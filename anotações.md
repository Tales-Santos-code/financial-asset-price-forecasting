# metrics
| Métrica | Descrição (O que é?) | Exemplo Prático (Ferrari - RACE) | Como Avaliar (Tá bom ou ruim?) |
| :--- | :--- | :--- | :--- |
| **MAE** <br>(Erro Absoluto) | É a média de quanto o modelo erra no dia a dia, ignorando o sinal (positivo/negativo). É a métrica mais estável e segura para treinar o modelo. | Se a ação subiu 3% e o modelo previu 2%, o MAE desse dia foi de 1%. | Quanto menor, melhor. Fica geralmente entre 0.01 (1%) e 0.02 (2%). É a sua bússola principal de treinamento. |
| **RMSE** <br>(Raiz do Erro Quadrático) | É o erro com "lente de aumento" para dias extremos. Ele pega os erros, eleva ao quadrado e depois tira a raiz. Ele grita quando o modelo erra feio. | Num dia atípico (Notícia de Guerra), a ação cai 8% e o modelo previu +1%. O RMSE "explode" para sinalizar esse erro grave. | Sempre será maior que o MAE. Use apenas para ver o quão vulnerável seu modelo é a "Cisnes Negros". Não tente zerar o RMSE. |
| **R² Score** <br>(Coeficiente de Determinação) | Mede a porcentagem do movimento da ação que o seu modelo conseguiu desvendar usando matemática. | O modelo olhou para o Lags, Bollinger e Volume e conseguiu entender 4.4% do porquê a ação se moveu. | Na faculdade, 80% é bom. Na bolsa de valores, entre 0.02 (2%) e 0.05 (5%) é SENSACIONAL. Se der 0.90 (90%), seu código tem erro de "Viagem no Tempo" (Data Leakage). |
| **Score CV vs Treino** <br>(Teste de Overfitting) | É a comparação entre a nota que o modelo tirou na prova em casa (Treino) vs na prova surpresa (CV/Teste). | Erro no Treino: 1.28%.<br>Erro no Teste (CV): 1.30%. | Os valores devem ser quase idênticos. Se o treino for 0.001 e o teste for 0.050, o modelo "decorou" o passado e vai falhar no futuro. |
| **Hit Rate** <br>(Acurácia Direcional) | É a métrica que paga as contas. Mede apenas se você acertou a cor da vela (Alta ou Baixa), ignorando o preço exato. | O modelo previu queda de -0.1%, mas a ação caiu -5.0%. O erro matemático foi gigante, mas o Hit Rate foi 100% (você lucrou na queda). | 50% é igual a jogar uma moeda. Modelos de fundos gigantes operam na faixa de 53% a 55%. Acima de 60% por muitos meses é raríssimo. |



# Experimento RaceFinancialPredict:

## RUN c91033a4d6014e0892036251e1a6b2f8:

aparentemente o modelo consegue prever se a ação vai subir ou descer mas não consegue prever o valor, acredito que adicionando features direcionais ira ajudar ele a prever o quanto pode subir e o quanto pode descer