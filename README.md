### CLIP-gradient-ascent-embeddings

- Generates matching text embeddings / a 'CLIP opinion' about images
- Uses gradient ascent to optimize text embeds for cosine similarity with image embeds
- Saves 'CLIP opinion' as .txt files [best tokens]
- Saves text-embeds.pt with [batch_size] number of embeds
- Can be used to create an adversarial text-image aligned dataset
- For XAI, adversarial training, etc; see 'attack' folder for example images
- Usage: Single image: `python gradient-ascent.py --use_image attack/024_attack.png`
- Usage: Batch process: `python gradient-ascent.py --img_folder attack`

-----
![gradient-ascent](https://github.com/user-attachments/assets/386645d8-5ed1-4799-9511-4ebe9746241c)
-----

Command-line arguments:

```
--batch_size', default=13, type=int, help="Reduce batch_size if you have OOM issues"
--model_name', default='ViT-L/14', help="CLIP model to use"
--tokens_to', type=str, default="texts", help="Save CLIP opinion texts path"
--embeds_to', type=str, default="embeds", help="Save CLIP embeddings path"
--use_best', type=str, default="True", help="If True, use best embeds (loss); if False, just saves last step (not recommended)"
--img_folder', type=str, default=None, help="Path to folder with images, for batch embeddings generation"
--use_image', type=str, default=None, help="Path to a single image"
```


Further processing example code snippets:

```
text_embeddings = torch.load("path/to/embeds.pt").to(device)

# loop over all batches of embeds and do a thing
num_embeddings = text_embeddings.size(0) # e.g. batch_size 13 -> idx 0 to 12
for selected_embedding_idx in range(num_embeddings):
    print(f"Processing embedding index: {selected_embedding_idx}")
    # do your thing here!


# select a random batch from embedding and do a thing
selected_embedding_idx = torch.randint(0, text_embeddings.size(0), (1,)).item()
selected_embedding = text_embeddings[selected_embedding_idx:selected_embedding_idx + 1].float()

# or just manually select one
selected_embedding_idx = 3
selected_embedding = text_embeddings[selected_embedding_idx:selected_embedding_idx + 1].float()
```

