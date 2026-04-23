# Integrantes
Leonardo de Matos Queiroz rm: 552500
Vitoria Cerqueira Santos rm: 552509
Lucas Henrique de Melo Silva rm: 550175
Pedro Henrique Sodre rm: 98834
Felipe Pereira Meschiatti rm: 551978


# Repositório Template para OpenCV

Este repositório serve como ponto a pé inicial para OpenCV em Python.

## Estrutura

- `requirements.txt` - lista de bibliotecas necessárias.
- `webcam.py` - script de exemplo que abre a câmera/video.
- `template.ipynb` - notebook com verificação básica da instalação do OpenCV.

## Configuração do ambiente

1. **Criar a virtualenv**:

   ```bash
   python3 -m venv venv
   ```

2. **Ativar o ambiente**:

   - macOS / Linux:

     ```bash
     source venv/bin/activate
     ```
   - Windows (PowerShell):

     ```powershell
     .\venv\Scripts\Activate.ps1
     ```

3. **Instalar as dependências**:

   ```bash
   pip install -r requirements.txt
   ```

## Testar a infraestrutura

- Rode o notebook `template.ipynb` para verificar a importação de `cv2` e `matplotlib` capturar um frame da webcam.

- Execute o script de exemplo:

  ```bash
  python webcam.py
  ```
  Uma janela mostrará a transmissão da câmera. Pressione `q` para sair.

