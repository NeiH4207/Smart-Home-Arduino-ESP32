import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from src.utils import MODEL_CLASSES, get_intent_labels, get_slot_labels, init_logger, load_tokenizer

import json
import random
import time
from collections import deque
from paho.mqtt import client as mqtt_client

client_id = 'mqtt_client_' + str(random.randint(0, 100))
username = 'emqx'
password = 'public'

broker = 'broker.emqx.io'
port = 1883

def parse_args():
    
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default=None, type=str, help="Input file for prediction")
    parser.add_argument("--input", default="xin chào", type=str, help="Input sentence for prediction")
    parser.add_argument("--output_file", default="output/results.csv", type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", 
                        default="/Users/mac/Desktop/IOT/BTL/Smart-Home-Intent-Detection-and-Slot-Filling/trained_models", 
                        type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", 
                        default="/Users/mac/Desktop/IOT/BTL/Smart-Home-Intent-Detection-and-Slot-Filling/BKAI", 
                        type=str, help="Path to save, load model")
    
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--use_rule_based", default=True, action="store_true", help="Rule for modify label")
    pred_config = parser.parse_args()
    return pred_config

logger = logging.getLogger(__name__)


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, "training_args.bin"))


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = MODEL_CLASSES[args.model_type][1].from_pretrained(
            args.model_dir, args=args, intent_label_lst=get_intent_labels(args), slot_label_lst=get_slot_labels(args)
        )
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except Exception:
        raise Exception("Some model files might be missing...")

    return model

def read_input_file(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    return lines

def convert_input_file_to_tensor_dataset(
    lines,
    pred_config,
    args,
    tokenizer,
    pad_token_label_id,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []
    all_real_len = []
    
    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[: (args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)
        all_real_len.append(len(tokens) + 2)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)
    all_real_len = torch.tensor(all_real_len, dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_slot_label_mask,
                            all_real_len)

    return dataset


class MqttClient(mqtt_client.Client):
    def __init__(self, client_id, username, password, broker, port):
        super().__init__(client_id)
        self.username_pw_set(username, password)
        self.connect(broker, port)
        self.received_messages = deque(maxlen=10)
        self.model = None
        
    def set_model(self, model):
        self.model = model
        
    def set_params(self, pred_config, args, tokenizer, pad_token_label_id):
        self.pred_config = pred_config
        self.args = args
        self.tokenizer = tokenizer
        self.pad_token_label_id = pad_token_label_id
        
        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)
        
        
    def inference(self, text):
        text = text.strip().lower().split()
        input_vec = convert_input_file_to_tensor_dataset([text], self.pred_config, self.args, 
                                                     self.tokenizer, self.pad_token_label_id)
        with torch.no_grad():
            input = {
                "input_ids": input_vec[0][0].reshape(1, -1),
                "attention_mask": input_vec[0][1].reshape(1, -1),
                "intent_label_ids": None,
                "slot_labels_ids": None,
                "real_lens": input_vec[0][-1].reshape(1, -1),
            }
            input["token_type_ids"] = input_vec[0][2].reshape(1, -1)
            outputs = self.model(**input)
            
            _, (intent_logits, slot_logits) = outputs[:2]
            intent_preds = intent_logits.detach().cpu().numpy()
            intent_preds = np.argmax(intent_preds, axis=1)
            
        # Slot prediction
        if self.args.use_crf:
            # decode() in `torchcrf` returns list with best index directly
            slot_preds = np.array(self.model.crf.decode(slot_logits))
        else:
            slot_preds = slot_logits.detach().cpu().numpy()
            
        all_slot_label_mask = input_vec[0][3].reshape(1, -1).detach().cpu().numpy()

        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)

        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

        for i in range(slot_preds.shape[0]):
            for j in range(slot_preds.shape[1]):
                if all_slot_label_mask[i, j] != self.pad_token_label_id:
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        for words, slot_preds, intent_pred in zip([text], slot_preds_list, intent_preds):
            line = ""
            line2 = ""
            for i in range(1, len(words)):
                if slot_preds[i][0] == 'B' and slot_preds[i - 1][0] == 'B' and \
                    slot_preds[i][2:] == slot_preds[i - 1][2:]:
                    slot_preds[i] = slot_preds[i].replace('B', 'I')
                if slot_preds[i - 1] == "B-devicedevice" and slot_preds[i] == "B-sysnumbersysnumber":
                    slot_preds[i] = "I-devicedevice"
            if intent_pred == 'greeting':
                slot_preds = ['O'] * len(words)
                
            for word, pred in zip(words, slot_preds):
                if pred == "O":
                    line = line + word + " "
                else:
                    line = line + "[{}:{}] ".format(word, pred)
                line2 = line2 + " {}".format(pred)
        
        if self.intent_label_lst[intent_preds[0]] == 'greeting':
            print("Pushing greeting")
            self.publish("iot/command_to_app", 
                         json.dumps({   
                            "command": "greeting",
                            "response": "Hi, I'm your assistant. How can I help you?"
                        }
                    )
                 )
        elif self.intent_label_lst[intent_preds[0]] == 'smart.home.check.status':
            self.publish("iot/command_to_app", 
                         json.dumps({   
                            "command": "check_status",
                            "response": "Okey!"
                        }
                    )
                 )
        elif self.intent_label_lst[intent_preds[0]] == 'smart.home.device.onoff':
            self.publish("iot/command_to_app", 
                         json.dumps({   
                            "command": "device_onoff",
                            "response": "Okey!"
                        }
                    )
                 )
        else:
            self.publish("iot/command_to_app", 
                         json.dumps({   
                            "command": "greeting",
                            "response": "Try again!"
                        }
                    )
                 )

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    def on_message(self, client, userdata, message):
        if message is None:
            return
        payload = str(message.payload.decode("utf-8"))
        print("Received message: %s" % payload)
        self.inference(payload)

def main():
    pred_config = parse_args()
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)
    
    if pred_config.use_rule_based:
        args.use_rule_based = True
    
    args.model_dir = pred_config.model_dir
    args.data_dir = pred_config.data_dir
    model = load_model(pred_config, args, device)
    if pred_config.use_rule_based:
        model.make_rule()
        model.args.use_rule_based = True
        
    logger.info(args)
    client = MqttClient(client_id, username, password, broker, port)
    client.set_model(model)
    
    pad_token_label_id = args.ignore_index
    tokenizer = load_tokenizer(args)
    client.set_params(pred_config, args, tokenizer, pad_token_label_id)
    
    client.subscribe("iot/command_to_ai")
    client.loop_forever()
        
if __name__ == "__main__":
    main()
