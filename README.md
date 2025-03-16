# MentalAI 
![Header](plots/header.png)  

MentalAI is a retrieval-augmented system for mental health support with fine-tuned suicide prediction. It leverages **llama.cpp**, reverse engineered **GPT2** and **LangChain** to provide AI-assisted conversations based on retrieved contextual information. The overall idea is to combine clinical expert knowledge with empathy and therapeutical experience by combining finetuned emotion and suicidality classifiers with academic, therapeutic and conversational knowledge documents.  

Here’s a more concise version of the contributions section, focusing on the tech stack:

## **Contributions**  

- **Kenneth Styppa**  
  - Developed the overall **RAG System** from data extraction, embedding, document retrieval and answer generation 
  - Curated **datasets A, B, C** and embedded subsets.  
  - Built **suicide risk prediction classifier** by reverse engineering and finetuning **GPT-2** from scratch.

- **Ole Plechinger**  
  - Added **emotion/sentiment classifier** to the pipeline
  - Embedded **full datasets** for system scaling.  
  - Conducted **system evaluations**.

- **Korhan Derin**  
  - Assisted with **scaling** and **evaluation**.
  - Performed **data analysis**.

## **Installation**  

### **1. Clone the repository**  
```bash
git clone https://github.com/kennethSty/mentalAI.git
cd mentalAI
```

### **2. Create and activate a Conda environment**  
```bash
conda create --name mentalai_env python=3.10  
conda activate mentalai_env  
```

### **3. Install dependencies**  
```bash
pip install -r requirements.txt  
```

### **4. Llama.cpp GPU installation:**  
Follow the latest instructions [here](https://python.langchain.com/docs/integrations/llms/llamacpp/#installation). 


### **5. Setup data and models**  
Place the folders available on [this google drive for the models](https://drive.google.com/drive/folders/13wcdsFVJpqAFZ9FG5u0nR--RTGCrlzU6?dmr=1&ec=wgc-drive-globalnav-goto) and this google drive for the data(TODO Ole: add the link to the data folder that includes the full chromastore)in the root directory. If your project root (like ours is called mental AI, place the folders within the mentalAI folder).
If links to the data folders are incomplete or outdated please contact [Ole Plechinger](mailto:ole.plechinger@protonmail.com).

### **6. Chatting with the bot** 
Go to `src/_2_chat/chat.py`, execute the script and chat away! 

```bash
python chat.py
```

## Optional reproduction of our set up

### **1. Rerun data extraction (optional)**
If you want to reproduce the data extraction process, go to `src/_0_data_preparation/extract_pubmed.py` and execute the script. 
Similarly run the `collect_counsel_datasets.py` and the `merge_counsel_datasets.py` files

### **2. Rerun data embedding (optional)**
If you want to reproduce the embedding process, go to `src/_1_chroma_preparation/doc2vec.py` and execute the script.

### **3. Rerun chroma upsert (optional)**
If you want to reproduce the embedding process, go to `src/_1_chroma_preparation/vec2chroma.py` and execute the script.

### **4. Rerun GPT2 finetuning (optional)**
If you want to reproduce the finetuning process, go to `src/_4_model_finetuning/finetuning.py` and execute the script.

### **5. Reproduce Finetuning evaluation GPT2 finetuning (optional)**
Make sure a finetuned model checkpoint exists in the models/finetuned directory. For the exact reproduction of our results, use `models/finetuned/gpt2_checkpoints/checkpoint_step_8000.pth`. Then, go to `src/_4_model_finetuning/finetuning.py` and execute the script.

### **6. Reproduce system evaluation (optional)**
TODO Ole: Add usage of the evaluation script


## Disclaimer
This project is intended for research and educational purposes. It is not a replacement for professional mental health services.

