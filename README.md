Projeto realizado para o programa de Trainee da Wise, ministrado pelos Tutores Rodrigo Quisen, Alberto Levi e Marcos Colodi.
A finalidade do projeto é analisar minhas habilidades em machine learning e ver o quão bem consigo aplicar o conteúdo ensinado através dos cursos.

(Descrição detalhada #TODO)

Descrição do projeto:
Após a validação de que o processo de tratamento de dados e geração de modelos de
classificação, o cliente solicitou a geração de uma API REST que permita o gerenciamento dos
processos de machine learning implementados. O objetivo é facilitar as interações com os modelos
gerados e também a realização de previsões em tempo real, de modo que todas as informações
sejam mantidas em um banco de dados MongoDB.

Requisitos


REQ-01 - Criar um diretório para organizar os arquivos do projeto, contendo:
● Um arquivo “README.md” como descritivo do projeto e as instruções para instalação;

● Um script python “main.py” para o processamento dos dados;

● Um arquivo “requirements.txt” para incluir as dependências do projeto.

REQ-02 - Versionar o diretório do projeto em um repositório privado no Github inserir os tutores
como colaboradores: quisen; mcolodi; AlbertoLevi.


REQ-03 - Inserir o dataset fornecido em uma collection chamada “CLIENTES” no banco.


REQ-04 - Implementar uma função para armazenar as previsões de churn realizadas no banco


(pode ser em uma collection a parte ou pode ser em um array dentro do documento do cliente).


REQ-05 - Implementar uma função para armazenar no banco de dados todas as informações dos
modelos gerados, bem como as métricas, hiperparâmetros, tempo de duração, datas, etc;


REQ-06 - Criar uma API REST com python utilizando Flask ou FastAPI;


REQ-07 - Implementar uma função que possibilite o carregamento de modelos em cache na API;


REQ-08 - Implementar um endpoint que recebe as features de um cliente e realize uma previsão de
churn em tempo real utilizando modelos em cache;


REQ-09 - Implementar um endpoint para consultar todas as previsões para um cliente específico;


REQ-10 - Implementar um endpoint para a listagem das previsões por faixas de probabilidade de
saída. Faixas sugeridas: [0%~25%], [25%~50%], [50%~75%], [75%~100%]. Como a busca poderá
retornar muitos registros, utilizar parâmetros para limitar a quantidade de registros retornados, bem
como parâmetros de offset que permitam a paginação dos resultados.