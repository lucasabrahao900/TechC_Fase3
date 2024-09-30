# Título do projeto

Com o objetivo de aprimorar o diagnóstico e o acompanhamento de pacientes com insuficiência cardíaca, desenvolvemos um modelo de classificação que combina análise de dados temporais com informações clínicas. Essa abordagem inovadora permite não apenas identificar pacientes em risco, mas também acompanhar a evolução da doença ao longo do tempo, otimizando o tratamento individualizado.

## 🚀 Começando

Essas instruções permitirão que você obtenha uma cópia do projeto em operação na sua máquina local para fins de desenvolvimento e teste.

### 📋 Pré-requisitos

De que coisas você precisa para instalar o software e como instalá-lo?

```bash
git clone https://github.com/lucasabrahao900/TechC_Fase3.git
```

Ao clonar o repositório, você terá acesso a tudo pertinente ao projeto, inclusive as bibliotecas necessárias através do arquivo requirements.txt. 

### 🔧 Instalação

Agora é necessário setar o ambiente para execução do projeto, recomendamos que utilize um ambiente virtual para o mesmo, logo

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS
```

Agora basta instalar as bibliotecas necessárias

```bash
pip install -r requirements.txt
```

## 📦 Implantação

Para executar apenas o app para testes, rode o seguinte comando:

```bash
streamlit run .\app.py
```

## 🛠️ Construído com

Mencione as ferramentas que você usou para criar seu projeto

* **Python:** Linguagem de programação principal.
* **Streamlit:** Framework para criação de aplicações web interativas.
* **Bibliotecas:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.