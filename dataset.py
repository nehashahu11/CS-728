# from datasets import load_dataset
# dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)
# # dataset.save_to_disk("conll-2003")
# train_data = dataset['train']

# 3. Print the first 10 instances
# print(f"{'Index':<6} | {'Tokens':<50} | {'NER Tags'}|{'Pos tags'}")
# print("-" * 80)

# for i in range(10):
#     tokens = train_data[i]['tokens']
#     tags = train_data[i]['ner_tags']
#     pos_tags = train_data[i]['pos_tags']
    
#     # Truncate tokens for cleaner display if they are too long
#     token_str = " ".join(tokens)[:47] + "..." if len(" ".join(tokens)) > 47 else " ".join(tokens)
    
#     print(f"{i:<6} | {token_str:<50} | {tags}| {pos_tags}")

from datasets import load_from_disk

# 1. Load the dataset from your local folder
dataset = load_from_disk("./conll-2003")
train_data = dataset['train']

# 2. Define the target Organization tag IDs
# In CoNLL-2003: 3 = B-ORG, 4 = I-ORG
org_tags = {3, 4}

print(f"Finding sentences with Organization tags...\n")

count = 0
for i, example in enumerate(train_data):
    # Check if any tag in the sentence's ner_tags list is in our org_tags set
    if any(tag in org_tags for tag in example['ner_tags']):
        count += 1
        sentence = " ".join(example['tokens'])
        
        # Optional: Highlight which tokens are organizations
        org_tokens = [example['tokens'][j] for j, t in enumerate(example['ner_tags']) if t in org_tags]
        
        print(f"[{count}] Sentence ID: {i}")
        print(f"Text: {sentence}")
        print(f"Organizations found: {org_tokens}")
        print("-" * 50)

print(f"\nTotal sentences with Organization tags: {count}")