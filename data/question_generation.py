from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
import json
from tqdm import tqdm
import jsonlines
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
# from transformers import T5ForConditionalGeneration, T5Tokenizer

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("./t5")
# hfmodel = T5ForConditionalGeneration.from_pretrained("./t5")

months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']

model_name = "./t5"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    # print(output)
    return output

# def run_model(input_string, **generator_args):
#     generator_args = {
#     "max_length": 256,
#     "num_beams": 4,
#     "length_penalty": 1.5,
#     "no_repeat_ngram_size": 3,
#     "early_stopping": True,
#     }
#     input_string = "generate questions: " + input_string + " </s>"
#     input_ids = tokenizer.encode(input_string, return_tensors="pt")
#     res = hfmodel.generate(input_ids, **generator_args)
#     output = tokenizer.batch_decode(res, skip_special_tokens=True)
#     output = [item.split("<sep>") for item in output]
#     return output


# run_model("shrouds herself in white and walks penitentially disguised as brotherly love through factories and parliaments; offers help, but desires power;")
# run_model("He thanked all fellow bloggers and organizations that showed support.")
# run_model("Races are held between April and December at the Veliefendi Hippodrome near Bakerky, 15 km (9 miles) west of Istanbul.")

origin_data = "./data/colloquial_claims/colloquial_claims_train.jsonl"
output_dir = "./data/colloquial_claims/"

def get_json_lines(inp_file):
    lines = []
    with jsonlines.open(inp_file) as reader:
        for obj in reader:
            lines.append(obj)
            
    return lines

def write_json_lines(output_file_name, list_data, output_folder):
    with jsonlines.open(output_folder+ output_file_name, mode='w') as writer:
        for dataline in list_data:
            writer.write(dataline)

data = get_json_lines(origin_data)
for i, datapoint in enumerate(tqdm(data)):
    # datapoint['dpr_evidence'] = retrieval_documents[i]
    datapoint['question'] = run_model(datapoint['colloquial_claims'][0])[0]

    # datapoint_text = datapoint['question'].split()
    # for i,d in enumerate(datapoint_text):
    #     if d.endswith('?'):
    #         datapoint['question_t5'] = ' '.join(datapoint_text[:i+1])
    #         break

write_json_lines("colloquial_claims_train_t5.jsonl", data, output_dir)

# TRANSFORMERS_OFFILNE=1 python main.py