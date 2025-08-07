In this Repo I've experimented with Decoder only Auto regressive text generation model, and Vision Transformer(Masted Auto Encoders) pre-training.

**The Language Model is Built following the GPT-2 Paper, Andrej Karpathy's Video Lectures** 

  **Size : 70 Million Parameters Approx**
  
  **Trained on Tiny Shakespeare Dataset on an RTX3050 for 5000 Steps with Batch size of 64**

  **The generated text is in data/generated.txt**

**I'm yet to train the ViT MAE Model as a single image forward pass requires atleast 3.8 GB of VRAM.**
**The goal is to pre-train the model on unlabelled images so it can pick up in common features in images, and then discard the decoder and add a classifier head to finetune the model.**
  
