# 🎓 Modelo Preditivo para Avaliação de Evasão Acadêmica
### Bacharelado em Sistemas de Informação - IFSULDEMINAS Campus Machado

Este projeto implementa uma proposta de Inteligência Artificial para identificar indícios de evasão escolar, auxiliando gestores na criação de estratégias antievasão personalizadas.

---

## 📝 Resumo do Projeto
A evasão acadêmica em cursos de tecnologia é um desafio que impacta o mercado de trabalho. Este trabalho utiliza **Machine Learning** para identificar as variáveis que explicam esse fenômeno, culminando em uma aplicação web preditiva.

* **Principais Algoritmos Testados:** Árvore de Decisão (AD), Floresta Randômica (FR), K-Vizinhos mais próximos (KNN), Support Vector Machine (SVM) e Naive Bayes.
* **Modelo Escolhido:** *Naive Bayes* (Acurácia de 67% e 76% de acertos para os casos de evasão).
* **Palavras-chave:** Inteligência Artificial, Análise de Dados Educacionais, Algoritmos de Aprendizagem.

---

## 🔒 Proteção de Dados e LGPD
Em conformidade com a **Lei Geral de Proteção de Dados (LGPD)**, este repositório **não contém a base de dados original** por envolver informações institucionais sensíveis. 
* Todos os resultados, capturas de tela e notebooks aqui apresentados utilizam **dados anonimizados** (variáveis demográficas e acadêmicas sem identificadores como Nome, CPF ou Matrícula). 
* O foco deste portfólio é a demonstração da metodologia de Ciência de Dados e a arquitetura da solução, garantindo a privacidade dos titulares dos dados.

---

## 📊 Análise e Visualização de Dados
Para o entendimento do cenário e exploração das variáveis (**Observação de Dados**), acesse o dashboard interativo:

🔗 **1. [Análise Geral dos Dados (Looker Studio)](https://lookerstudio.google.com/reporting/5fddd255-7cb8-49c5-b79e-dcacf117c4f4)**

---

## 🔗 Repositórios e Aplicações

Abaixo estão os links diretos para as frentes de desenvolvimento e códigos-fonte do projeto:

* **1. Notebooks:** [Visualizar Notebooks](https://drive.google.com/drive/folders/1FyKA6ylhf3Z_hR-XsyIfWfPhDpJB5kWK?usp=drive_link)
* **2. Backend (API Flask):** [projetoClassificacao](https://github.com/sanderpiva/projetoClassificacao.git)
* **3. Frontend (React):** [reactClassificacaoTCC](https://github.com/sanderpiva/reactClassificacaoTCC.git)
* **4. Aplicação Web Final:** [🚀 Disponível na Vercel](https://react-vercel-classificacao.vercel.app/)

---
## ⚠️ Alternativas de Acesso e Execução
Caso a aplicação web esteja temporariamente offline, utilize uma das opções abaixo para executar o sistema:

**1. GitHub Codespace**  
Acesse o repositório [`tcc-evasao-codespace`](https://github.com/sanderpiva/tcc-evasao-codespace.git) e execute a aplicação diretamente no navegador. O ambiente já vem configurado com todas as dependências, sem necessidade de instalação local.

**2. Flutter**  
Execute a aplicação utilizando o framework Flutter através de uma das seguintes formas:

- **Ambiente Local**: Instale o Flutter SDK em [flutter.dev](https://flutter.dev) e execute o arquivo `tcc_flutter.zip` em seu computador. Permite rodar a aplicação via mobile ou desktop.

- **FlutLab Online**: Para teste rápido sem instalações, importe o arquivo `tcc_flutter.zip` diretamente no navegador através do [flutlab.io](https://flutlab.io).

---

## 📂 Código fonte da aplicação em Flutter

Para garantir a transparência e facilitar a revisão técnica, o projeto pode ser:

* **1. Visualizado Diretamente:** Acesse a pasta [**/src**](./src) para navegar pelo código-fonte (`.dart`) e arquivos de configuração. 
   > *Nota: Nesta versão, a classe anteriormente rotulada como **"Matriculado"** foi renomeada para **"Não Evadido"** (Classe 0) para representar com maior precisão os estudantes que permanecem no curso ou já concluíram, diferenciando-os dos casos de evasão efetiva.*

---
---

## 📈 Análise de Melhorias e Roadmap Técnico

Como parte do processo de pós-defesa e melhoria contínua deste portfólio, identifiquei oportunidades de otimização que podem ser aplicadas para elevar ainda mais a performance do modelo, especificamente em relação ao desbalanceamento observado entre as classes de "Evasão" e "Não Evadido":

* **Tratamento de Desbalanceamento**: O modelo atual utiliza uma abordagem *baseline* (baseada na distribuição original dos dados). Para otimizar a sensibilidade (*recall*) na identificação da classe minoritária (casos de evasão), a implementação de técnicas como o **SMOTE** (para criação de exemplos sintéticos) ou a aplicação de **class_weights='balanced'** (em modelos como RandomForest ou Regressão Logística) seriam as próximas etapas naturais de evolução.
* **Refinamento do Modelo**: Embora o *Naive Bayes* tenha apresentado uma performance satisfatória para a proposta inicial de pesquisa, a transição para algoritmos baseados em *Ensemble* (como *XGBoost* ou *Random Forest*) com ajuste de hiperparâmetros (via *GridSearch* ou *RandomizedSearch*) permitiria explorar fronteiras de decisão mais robustas.
* **Transparência**: Este repositório mantém o código original submetido na defesa do TCC. O objetivo de manter estas observações é documentar o aprendizado contínuo e oferecer um caminho claro para futuras iterações deste projeto.

---

## 👥 Autoria e Orientação

* **Autor:** Sander Gustavo Piva
* **Orientador:** Matheus Eloy Franco

---

## 📸 Demonstração da Aplicação

### Interface Flask (Testes de Predição)
<br>

**Teste: Indicação de EVASÃO**
<br>

![Foto 1: Teste EVASAO](https://github.com/sanderpiva/TCC_Final/blob/main/imgs/flask_evade.png?raw=true)

<br>

**Teste: Indicação de NÃO EVASÃO**
<br>

![Foto 2: Teste NAO EVASAO](https://github.com/sanderpiva/TCC_Final/blob/main/imgs/flask_nao_evade.png?raw=true)

<br>

### Interface Final em React (Vercel)
<br>

![Foto 3: App no VERCEL](https://github.com/sanderpiva/TCC_Final/blob/main/imgs/react.png?raw=true)

<br>

**Teste no Flutter: Indicação de Risco de EVASÃO**

<br>

![Foto 4: App em Flutter](https://github.com/sanderpiva/TCC_Final/blob/main/imgs/tcc_flutter.jpg?raw=true)

---

#### 🚧 NoDropOut Soft 🚀 Finalizado: Out/2024 | Versão atualizada: Abr/2026
