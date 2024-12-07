import warnings
warnings.filterwarnings('ignore') # Disable spam
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import os
import clip
import kornia
import torch
from safetensors.torch import load_file
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from colorama import Fore, Style
import copy
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1', 'yes'}:
        return True
    elif value.lower() in {'false', '0', 'no'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='CLIP gradient ascent')
    parser.add_argument('--batch_size', default=13, type=int, help="Reduce batch_size if you have OOM issues")
    parser.add_argument('--model_name', default='ViT-L/14', help="CLIP model to use")
    parser.add_argument('--tokens_to', type=str, default="texts", help="Save CLIP opinion texts path")
    parser.add_argument('--embeds_to', type=str, default="embeds", help="Save CLIP embeddings path")
    parser.add_argument('--use_best', type=str, default="True", help="If True, use best embeds (loss); if False, just saves last step (not recommended)")
    parser.add_argument('--img_folder', type=str, default=None, help="Path to folder with images, for batch embeddings generation")
    parser.add_argument('--use_image', type=str, default=None, help="Path to a single image")
    return parser.parse_args()

# CLIP Model Loader
def load_clip_model(model_name, device):
    # Check if `model_name` is a valid file path
    if os.path.exists(model_name):
        print(f"Loading model from path: {model_name}")
        
        if model_name.endswith(".safetensors"):
            print(f"Detected .safetensors file; assuming ViT-L/14 for loading.")
            model, preprocess = clip.load("ViT-L/14", device)
            state_dict = load_file(model_name)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model = model.eval().float()
        else:
            _, preprocess = clip.load("ViT-L/14", device)
            model = torch.load(model_name).to(device).float()
            model = model.eval().float()
    else:
        available_models = clip.available_models()
        if model_name in available_models:
            print(f"Detected OpenAI/CLIP model: {model_name}")
            model, preprocess = clip.load(model_name, device)
            model = model.eval().float()
        else:
            raise ValueError(Fore.RED + Style.BRIGHT + f"\n\nInvalid model_name: '{model_name}'. Must be a file path or one of {available_models}." + Fore.RESET)
    
    return model, preprocess

# Image Loader
def load_image(img_path, sideX, sideY):
    im = torch.tensor(np.array(Image.open(img_path).convert("RGB"))).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255
    im = F.interpolate(im, (sideX, sideY))
    return im

# Normalization
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(-1, 1, 1))

    def forward(self, img):
        return (img - self.mean) / self.std

# Augmentation Pipeline
def augment(into, augs):
    return augs(into)

# Gradient Ascent Functions, originally implemented by- X: @advadnoun
def clip_encode_text(model, text, many_tokens, prompt):
    x = torch.matmul(text, model.token_embedding.weight)
    x = x + model.positional_embedding
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x)
    x = x[torch.arange(x.shape[0]), many_tokens + len(prompt) + 2] @ model.text_projection
    return x

# Entertain user by printing CLIP's 'opinion' rants about image to console
def checkin(loss, tx, lll, tok, bests, imagename, tokens_to):
   
    unique_tokens = set()

    these = [tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist()).replace('', '').replace('', '') for kj in range(lll.shape[0])]
    
    for kj in range(lll.shape[0]):
        if loss[kj] < sorted(list(bests.keys()))[-1]:
            cleaned_text = ''.join([c if c.isprintable() else ' ' for c in these[kj]])
            bests[loss[kj]] = cleaned_text
            bests.pop(sorted(list(bests.keys()))[-1], None)
            try:
                decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist())
                decoded_tokens = decoded_tokens.replace('<|startoftext|>', '').replace('<|endoftext|>', '') # Don't spam console with CLIP's special tokens.
                decoded_tokens = ''.join(c for c in decoded_tokens if c.isprintable())
                print(Fore.WHITE + f"Sample {kj} Tokens: ")
                print(Fore.BLUE + Style.BRIGHT + f"{decoded_tokens}" + Fore.RESET)
            except Exception as e:
                print(f"Error decoding tokens for sample {kj}: {e}")
                continue

    for j, k in zip(list(bests.values())[:5], list(bests.keys())[:5]):
        j = j.replace('<|startoftext|>', '')
        j = j.replace('<|endoftext|>', '')
        j = j.replace('\ufffd', '') # Also remove other characters that can cause problems, if txt files are used for whatever downstream tasks
        j = j.replace('.', '')
        j = j.replace(';', '')
        j = j.replace('?', '')
        j = j.replace('!', '')
        j = j.replace('_', '')
        j = j.replace('-', '')
        j = j.replace('\\', '')
        j = j.replace('\'', '')
        j = j.replace('"', '')
        j = j.replace('^', '')
        j = j.replace('&', '')
        j = j.replace('#', '')
        j = j.replace(')', '')
        j = j.replace('(', '')
        j = j.replace('*', '')
        j = j.replace(',', '')
        tokens = j.split()
        unique_tokens.update(tokens)

    with open(f"{tokens_to}/{imagename}.txt", "w", encoding='utf-8') as f:
        f.write(" ".join(unique_tokens))

# Softmax
class Pars(torch.nn.Module):
    def __init__(self, batch_size, many_tokens, prompt):
        super(Pars, self).__init__()
        self.batch_size = batch_size
        self.many_tokens = many_tokens
        self.prompt = prompt
        
        # Initialize parameters
        st = torch.zeros(batch_size, many_tokens, 49408).normal_()
        self.normu = torch.nn.Parameter(st.cuda())
        self.much_hard = 1000

        self.start = torch.zeros(batch_size, 1, 49408).cuda()
        self.start[:, :, 49406] = 1

        self.prompt_embeddings = torch.zeros(batch_size, len(prompt), 49408).cuda()
        for jk, pt in enumerate(prompt):
            self.prompt_embeddings[:, jk, pt] = 1 
        
        self.update_padding()

    def update_padding(self):
        # Update the padding tokens based on current number of active tokens
        pad_length = 77 - (self.many_tokens + len(self.prompt) + 1)
        self.pad = torch.zeros(self.batch_size, pad_length, 49408).cuda()
        self.pad[:, :, 49407] = 1

    def diversity_penalty(self, new_tokens, existing_tokens, min_sim=0.6, max_sim=0.9):
        #Penalize new tokens for being too similar (>max_sim) or too dissimilar (<min_sim) to existing tokens.

        # Compute cosine similarity between new tokens and existing tokens
        cosine_sim = F.cosine_similarity(new_tokens.unsqueeze(1), existing_tokens, dim=-1)

        # Identify where similarity is outside the acceptable range
        too_similar = (cosine_sim > max_sim).float()
        too_dissimilar = (cosine_sim < min_sim).float()

        # Penalize both cases
        penalty = too_similar * (cosine_sim - max_sim) ** 2  # Penalty for being too similar
        penalty += too_dissimilar * (min_sim - cosine_sim) ** 2  # Penalty for being too dissimilar

        # Return the mean penalty
        return penalty.mean()

    def add_tokens(self, num_new_tokens, model, image, optimizer, prompt, many_tokens, nom, augment):
        # Add more tokens with refined gradient-based initialization during GA
        
        # Compute gradients for the current tokens
        loss, _, _ = ascend_txt(image, model, self, many_tokens, prompt, nom, augment)
        loss = loss.mean()  # Mean over the batch
        loss.backward()  # Compute gradients
        gradients = self.normu.grad  # Gradients w.r.t. current tokens

        # Weight gradients by their norm
        gradient_weights = gradients.norm(dim=-1, keepdim=True)  # Compute gradient magnitudes
        weighted_gradients = gradients * gradient_weights  # Scale gradients by magnitude
        weighted_mean = weighted_gradients.mean(dim=1, keepdim=True)  # Compute weighted mean

        # Use the weighted gradient mean to initialize new tokens
        new_tokens = weighted_mean.repeat(1, num_new_tokens, 1)
        new_tokens += torch.normal(mean=0, std=0.01, size=new_tokens.shape).cuda()

        # Apply diversity penalty to ensure new tokens are distinct but related
        existing_tokens = self.normu  # Existing token embeddings
        penalty = self.diversity_penalty(new_tokens, existing_tokens)
        new_tokens -= penalty * 0.1  # Adjust tokens based on penalty weight

        # Update normu with the new tokens
        self.normu = torch.nn.Parameter(torch.cat([self.normu, new_tokens], dim=1))
        self.many_tokens += num_new_tokens
        self.update_padding()

    def forward(self):
        self.soft = F.gumbel_softmax(self.normu, tau=self.much_hard, dim=-1, hard=True)
        fin = torch.cat([self.start, self.prompt_embeddings, self.soft, self.pad], 1)
        return fin



# Gradient Ascent
def ascend_txt(image, model, lats, many_tokens, prompt, nom, augment):
    iii = nom(augment(image[:,:3,:,:].expand(lats.normu.shape[0], -1, -1, -1)))
    iii = model.encode_image(iii).detach()
    lll = lats()
    tx = clip_encode_text(model, lll, many_tokens, prompt)
    return -100 * torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, lats.normu.shape[0]).T.mean(1), tx, lll

# Optimization loop with AMP (Automatic Mixed Precision)
def train(image, model, lats, many_tokens, prompt, optimizer, nom, augment):
    with autocast():
        loss1, tx, lll = ascend_txt(image, model, lats, many_tokens, prompt, nom, augment)
    loss = loss1.mean()
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss1, tx, lll

# Obtain the best-loss text embeds - or else, just save last state (even if worse; not recommended)
def generate_target_text_embeddings(img_path, model, lats, optimizer, scheduler, training_iterations, checkin_step, many_tokens, prompt, nom, augment, tok, bests, tokens_to, embeds_to, use_best):

    if use_best:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = load_image(img_path, model.visual.input_resolution, model.visual.input_resolution)
        print(Fore.YELLOW + Style.BRIGHT + f"\nRunning gradient ascent for {img_name}...\n" + Fore.RESET)

        best_loss = float('inf')  # Initialize the best loss as infinity
        best_text_embeddings = None  # Placeholder for the best text embeddings

        for j in range(training_iterations):
            # Adjust active tokens dynamically at specific steps
            if j == 50:
                num_new_tokens = 1 # Add one more token for CLIP to optimize after some initial convergence
                print(Fore.YELLOW + Style.BRIGHT + f"Adding {num_new_tokens} tokens at step {j}..." + Fore.RESET)
                lats.add_tokens(num_new_tokens, model, img, optimizer, prompt, many_tokens, nom, augment)
                
                # Reinitialize the optimizer and scheduler with updated parameters
                optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.8)

            # Training step
            loss, tx, lll = train(img, model, lats, many_tokens, prompt, optimizer, nom, augment)
            current_loss = loss.mean().item()
            
            # Update best embeddings if current loss is better
            if current_loss < best_loss:
                best_loss = current_loss
                best_text_embeddings = copy.deepcopy(tx.detach())
                print(Fore.RED + Style.BRIGHT + f"New best loss: {best_loss:.3f}" + Fore.RESET)
                checkin(loss, tx, lll, tok, bests, img_name, tokens_to)
                print(Fore.RED + Style.BRIGHT + "-------------------" + Fore.RESET)
            
            scheduler.step()

            # Print learning rate for monitoring
            if j % checkin_step == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(Fore.GREEN + f"Iteration {j}: Average Loss: {current_loss:.3f}" + Fore.RESET)
                checkin(loss, tx, lll, tok, bests, img_name, tokens_to)
        
        # Save the best embeddings
        torch.save(best_text_embeddings, f"{embeds_to}/{img_name}.pt")
        print(Fore.MAGENTA + Style.BRIGHT + f"\nBest text embedding saved to '{embeds_to}'.\nTokens (CLIP 'opinion') saved to '{tokens_to}'.\n" + Fore.RESET)
        
        del optimizer, scheduler, lats, bests, prompt
        return img, best_text_embeddings, img_path

    else:     
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = load_image(img_path, model.visual.input_resolution, model.visual.input_resolution)
        print(Fore.YELLOW + Style.BRIGHT + f"\nRunning gradient ascent for {img_name}...\n" + Fore.RESET)
        for j in range(training_iterations):
            loss, tx, lll = train(img, model, lats, many_tokens, prompt, optimizer, nom, augment)
            if j % checkin_step == 0:
                print(Fore.GREEN + f"Iteration {j}: Average Loss: {loss.mean().item()}" + Fore.RESET)
                checkin(loss, tx, lll, tok, bests, img_name, tokens_to)
       
        target_text_embedding = tx.detach()
        torch.save(target_text_embedding, f"{embeds_to}/{img_name}.pt")
        print(Fore.MAGENTA + Style.BRIGHT + f"\nBest text embedding saved to '{embeds_to}'.\nTokens (CLIP 'opinion') saved to '{tokens_to}'.\n" + Fore.RESET)
        
        return img, target_text_embedding, img_path



# Main loop
def main():
    args = parse_arguments()

    # Graceful argument validation
    if not args.img_folder and not args.use_image:
        print(Fore.RED + "Error: No image source specified." + Fore.RESET)
        print(Fore.YELLOW + "Please specify path to either --img_folder for batch processing, or --use_image for single-image processing!" + Fore.RESET)
        return

    use_best = str2bool(args.use_best)
    tokens_to = args.tokens_to
    embeds_to = args.embeds_to
    os.makedirs(tokens_to, exist_ok=True)
    os.makedirs(embeds_to, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
            
    normalizer = Normalization([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]).cuda()
                
    model, preprocess = load_clip_model(args.model_name, device)
            
    tok = clip.simple_tokenizer.SimpleTokenizer()
    bests = {1000: 'None', 1001: 'None', 1002: 'None', 1003: 'None', 1004: 'None', 1005: 'None'}
    prompt = clip.tokenize('''''').numpy().tolist()[0]
    prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]
                
    lats = Pars(args.batch_size, 4, prompt).cuda()
               
    augs = torch.nn.Sequential(
        kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda(),
    ).cuda()

    optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.8)

    if args.img_folder is not None:
        try:
            image_folder = args.img_folder
        except Exception as e:
                print(f"Please specify a valid '--img_folder /path/myfolder': {e}") 
 
        # Get all valid image files in the folder
        valid_extensions = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')
        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                       if f.lower().endswith(valid_extensions)]
        print(f"\n --- Batch processing {len(image_files)} images. ---")
        for img_path in image_files:
            try:
                tok = clip.simple_tokenizer.SimpleTokenizer()
                bests = {1000: 'None', 1001: 'None', 1002: 'None', 1003: 'None', 1004: 'None', 1005: 'None'}
                prompt = clip.tokenize('''''').numpy().tolist()[0]
                prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]
                
                lats = Pars(args.batch_size, 4, prompt).cuda()
               
                augs = torch.nn.Sequential(
                    kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda(),
                ).cuda()                
                optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.8)

                
                img, target_text_embedding, img_path = generate_target_text_embeddings(img_path, model, lats, optimizer, scheduler, 340, 10, 4, prompt, normalizer, augs, tok, bests, tokens_to, embeds_to, use_best)
                print(f"Done processing image: {img_path}")
            except Exception as e:
                print(f"Something went wrong: {e}")

    else:
        try:
            lats = Pars(args.batch_size, 4, prompt).cuda()              
               
            tok = clip.simple_tokenizer.SimpleTokenizer()
            bests = {1000: 'None', 1001: 'None', 1002: 'None', 1003: 'None', 1004: 'None', 1005: 'None'}
            prompt = clip.tokenize('''''').numpy().tolist()[0]
            prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]
                
            lats = Pars(args.batch_size, 4, prompt).cuda()
               
            augs = torch.nn.Sequential(
                kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda(),
            ).cuda()                

            optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.8)                
            
            img, target_text_embedding, img_path = generate_target_text_embeddings(args.use_image, model, lats, optimizer, scheduler, 340, 10, 4, prompt, normalizer, augs, tok, bests, tokens_to, embeds_to, use_best)
            print(f"Done processing image: {img_path}")
        except Exception as e:
            print(f"Something went wrong: {e}")
    
if __name__ == "__main__":
    main()
