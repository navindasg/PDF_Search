# Steps to Run

## Ollama
For this project I am using all local models. For embedding I am using Nomic-Text-Embed, for Vision I am using llava:13b, and for the LLM I am using gemma3:17b (highest ranking open source model across berkely LLM rankings as of 3/18/25) or phi4.

### Space requirements:
* Nomic :                   ~300 MB
* llava :                   ~8 GB
* gemma3:17b :              ~18 GB
* phi4 (optional) :         ~10 GB

### Specs:
Running using: Apple Silicon M3 Max / 32 GB RAM.
Reccomended: 16 Core Processer / 64 GB RAM.

### How to Download
- Go to ollama.com and download ollama
- Go to terminal and run:
```console
ollama pull nomic-embed-text
ollama pull llava:13b
ollama pull gemma3:27b
```
***Optional***
```console
ollama pull phi4
```
 
## Python Environment
Currently I have not set up a UI for this project so you will need to set up a python environment. This is running on Python 3.13.1.
### Steps for Environment Setup
- Create a .venv running python 3.13.1
- Activate the .venv:
#### Mac: in the terminal run:
```console
cd .venv
source bin/activate
```
#### Windows: navigate to .venv/bin/Activate.ps1 and run it.

## Requirements
In order to install the python requirements go into terminal and run:
```console
pip install -r requirements.txt
```

## Running the Code
- Place all of the desired PDFs in the folder pdf_input (Some testing files are in there and can be removed)
- Run pdf_parser.py
- Run main.py

## Future Steps
- I plan to do some more advanced testing and according prompt tuning.
- I also plan to add a UI that lets you choose between models.
- Data tracing (LLM output to PDF image/text/table) is also another feature I would want to add to improve functinality.
