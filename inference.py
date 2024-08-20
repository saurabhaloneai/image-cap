import torch
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from src.model import EncoderDecoder

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#  model
embed_size = 300
attention_dim = 256
encoder_dim = 2048
decoder_dim = 512
drop_prob = 0.3

model = EncoderDecoder(
    embed_size=embed_size,
    attention_dim=attention_dim,
    encoder_dim=encoder_dim,
    decoder_dim=decoder_dim,
    drop_prob=drop_prob
).to(device)

# load the pre-trained model weights
model.load_state_dict(torch.load("/model_weights.pth", map_location=device))
model.eval()

# image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_caption(image_path, model, tokenizer, max_len=50):
    # load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # encode the image
    features = model.encoder(image_tensor)
    
    # initialize the hidden and cell states
    h, c = model.decoder.init_hidden_state(features)
    
    # start the caption with the [CLS] token
    word = torch.tensor([tokenizer.cls_token_id]).to(device)
    embeds = model.decoder.embedding(word)
    
    captions = []
    alphas = []
    
    for _ in range(max_len):
        alpha, context = model.decoder.attention(features, h)
        alphas.append(alpha.cpu().detach().numpy())
        
        lstm_input = torch.cat((embeds.squeeze(1), context), dim=1)
        h, c = model.decoder.lstm_cell(lstm_input, (h, c))
        
        output = model.decoder.fcn(model.decoder.drop(h))
        predicted_word_idx = output.argmax(dim=1)
        
        captions.append(predicted_word_idx.item())
        
        # break if [SEP] token is generated
        if predicted_word_idx.item() == tokenizer.sep_token_id:
            break
        
        embeds = model.decoder.embedding(predicted_word_idx.unsqueeze(0))
    
    # convert word indices to words, skipping special tokens
    caption = tokenizer.decode(captions, skip_special_tokens=True)
    return image, caption

# test

image_path = "/new-test-data2/600-07653932en_Masterfile.jpg" 
image, caption = predict_caption(image_path, model, tokenizer)
plt.imshow(image)
plt.title("generated caption: " + caption)
plt.axis("off")
plt.show()
