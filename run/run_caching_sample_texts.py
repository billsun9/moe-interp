# DEPRECATED
from moe.activation_extraction.run_caching_utils import process_sample_texts
from moe.activation_extraction.activation_extractor_utils import OLMoEActivationExtractor
from datetime import datetime
MODEL_NAME = "allenai/OLMoE-1B-7B-0125-Instruct"
MODEL_CACHE_DIR = "/local/bys2107/hf_cache"
extractor = OLMoEActivationExtractor(model_name = MODEL_NAME, cache_dir = MODEL_CACHE_DIR)
extractor.load_model_and_tokenizer()
# ---------------------------------------------------
# dataset_name = "fluffy_or_pointy_objects"
# sample_texts = [
#     "pillow cotton candy soft fluffy bed cushion",
#     "bunny fleece jackets bathrobes plush blanket marshmallow",
#     "spear knife serrated sharp pointy",
#     "cactus dart arrowhead fishing hook spiky"
# ]
# metadata = {
#     "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     "desc": "testing on strings of fluffy objects"
# }

# save_path = process_sample_texts(sample_texts, extractor, dataset_name, metadata=metadata)

# data = extractor.load_activations(save_path)

# print(data.keys())
# print(data['metadata'])
# extractor.print_activation_summary(data)
# ---------------------------------------------------
# dataset_name = "random_strings"
# sample_texts = [
#     "orbit mirror canyon velvet cloud brisk meadow lantern twist",
#     "echo drift summit pine galaxy rustle amber token willow spark flame",
#     "crystal dune maple thunder hatch comet reef fabric tunnel glow quartz",
#     "ladder ember frost canvas breeze cliff pebble static ivy"
# ]
# metadata = {
#     "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     "desc": "testing on random strings"
# }

# save_path = process_sample_texts(sample_texts, extractor, dataset_name, metadata=metadata)

# data = extractor.load_activations(save_path)

# print(data.keys())
# print(data['metadata'])
# extractor.print_activation_summary(data)
# ---------------------------------------------------
dataset_name = "simple_math"
sample_texts = [
    "calculate twenty two times eleven",
    "what's the square root of nineteen",
    "I was born on June tenth, my sister was born on July second",
    "find the slope of this equation: 3y = 2x + 10"
]
metadata = {
    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "desc": "simple math questions"
}
save_path = process_sample_texts(sample_texts, extractor, dataset_name, metadata=metadata)

data = extractor.load_activations(save_path)

print(data.keys())
print(data['metadata'])
extractor.print_activation_summary(data)

