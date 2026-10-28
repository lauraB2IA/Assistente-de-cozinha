# Roteiro do relatório

## Título
Reconhecimento de comandos de voz para um assistente de cozinha usando redes neurais

## Objetivo
Desenvolver um programa em Python capaz de identificar comandos de voz curtos e associá-los às etapas de uma receita, permitindo iniciar a receita, avançar para o próximo passo, repetir uma instrução ou finalizar o preparo sem tocar no celular.

## Metodologia
Foram coletados áudios próprios organizados em pastas por classe. A biblioteca Librosa foi usada para carregar os arquivos WAV, padronizar a frequência de amostragem e extrair MFCCs. Também foram calculadas as características delta e delta-delta, que representam variações temporais do áudio.

Os dados foram divididos em treino e teste. O modelo foi uma rede neural convolucional (CNN) criada com TensorFlow/Keras. As camadas convolucionais aprendem padrões presentes no espectrograma/MFCC; as camadas de pooling reduzem a dimensão; dropout ajuda a diminuir overfitting; e a camada Softmax produz a probabilidade de cada comando.

## Classes
`iniciar_receita`, `proximo_passo`, `repetir_instrucao`, `finalizar_receita` e `desconhecido`.

## Avaliação
A avaliação deve ser feita no conjunto de teste, que não é usado para atualizar os pesos da rede. Registre neste trecho os valores exibidos pelo programa:

- Acurácia: ____
- Precision/Recall/F1 de `iniciar_receita`: ____
- Precision/Recall/F1 de `proximo_passo`: ____
- Precision/Recall/F1 de `repetir_instrucao`: ____
- Precision/Recall/F1 de `finalizar_receita`: ____
- Precision/Recall/F1 de `desconhecido`: ____

Inclua `resultados/matriz_confusao.png` e explique quais classes foram mais confundidas. Inclua também `resultados/curvas_treinamento.png` para discutir convergência e possível overfitting.

## Conclusão sugerida
O projeto demonstra que características acústicas extraídas com Librosa podem ser usadas por uma CNN para reconhecer comandos de voz em um assistente de cozinha. A qualidade do resultado depende da quantidade e variedade do dataset, do ruído do ambiente e da semelhança entre as frases. Como melhoria, podem ser usadas mais gravações, aumento de dados com ruído controlado e integração das previsões com um aplicativo que mostre o passo atual da receita.
