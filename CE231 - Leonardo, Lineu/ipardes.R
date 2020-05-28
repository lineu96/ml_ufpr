#.............................................................................
# MACHINE LEARNING

#.............................................................................

# Lineu Alberto C. de Freitas - GRR20149144
# Leonardo Henrique B. Kruger - GRR20149101

#.............................................................................
# ENCODING: ISO-8859-10
#.............................................................................

# CONTEXTO

# Dados coletados via consulta pública no site do IPARDES
# http://www.ipardes.pr.gov.br/imp/imp.php

# Objetivo - classificar, com base em preditoras, um município em duas classes de acordo com o balanço final da cidade.
# o cálculo feito foi: renda do município - gastos do município; este resultado foi dicotomizado da seguinte forma:

# ok    - balanço final acima da mediana
# risco - balanço final abaixo da mediana

#.............................................................................

# VARIÁVEIS

# cid - cidade
# ano
# aaua - Abastecimento de Água - Unidades Atendidas
# nvt - Nascidos Vivos - Total
# abt -  Agências Bancárias - Total 
# eec - Energia Elétrica - Consumo (Mwh)
# cmp - Crianças Menores de 2 anos Pesadas 
# pibpc - Produto Interno Bruto per Capita (R$ 1
# att - Acidentes de Trânsito - Total 
# mert - Matrículas no Ensino Regular - Total
# dd - Densidade Demográfica (hab/km²)
# at - Área Territorial (km²)
# dsmc - Distância da Sede Municipal à Capital (km) 
# fvt - Frota de Veículos - Total 
# meit - Matrículas na Educação Infantil - Total 
# mct - Matrículas na Creche - Total 
# cavt - Consumo de Água - Volume Faturado (m3)
# mpet - Matrículas na Pré-Escola - Total
# vhd - Vítimas de Homicídio Doloso 
# meft - Matrículas no Ensino Fundamental - Total
# lat - Vítimas de Roubo com Resultado de Morte (Latrocínio) 
# vlc - Vítimas de Lesão Corporal com Resultado de Morte 
# vhct - Vítimas de Homicídio Culposo no Trânsito 
# mem - Matrículas no Ensino Médio - Total 
# ipdm - Índice Ipardes de Desempenho Municipal (IPDM) 
# mep - Matrículas na Educação Profissional - Total 
# hiv - Número de Casos por HIV / AIDS - Total 
# obit - Óbitos (CID10) - Total (Mortalidade Geral)
# bcg - Cobertura Vacinal - BCG (Tuberculose) (%)
# hepa - Cobertura Vacinal - Hepatite A (HA) (%)
# hepb - Cobertura Vacinal - Hepatite B (HB) (%) 
# poli - Cobertura Vacinal - Poliomielite (VOP) (%)
# fa - Cobertura Vacinal - Febre Amarela (FA) (%) 
# rota - Cobertura Vacinal - Rotavírus Humano (VORH) (%) 
# meni - Cobertura Vacinal - Meningocócica Conjugada (Men C) (%) 
# pne - Cobertura Vacinal - Pneumocócica 10V (Pncc10V) (%) 
# tri - Cobertura Vacinal - Tríplice Viral (SCR) (%)
# tet - Cobertura Vacinal - Tetra Viral (SCR+VZ) (%)
# dtp - Cobertura Vacinal - Tríplice (DTP)
# tpb -  Tetra / Penta Bacteriana (DTP+Hib+HB) (%)
# pent - Cobertura Vacinal - Penta Bacteriana (Pentavalente) (DTP+Hib+HB) (PENTA) (%)
# papi - Cobertura Vacinal - Papilomavírus Humano (HPV) (%)
# cvdt - Cobertura Vacinal - Dupla Adulto (dT) e Tríplice Acelular Gestante (dTpa) (%)
# cvta - Cobertura Vacinal - Tríplice Acelular Gestante (dTpa) (%)
# aaq - Aeroportos e Aeródromos - Quantidade 

# rm - Receitas Municipais - Total 
# dmt - Despesas Municipais - Total 

# saldo - (rm - dmt)

# class - risco (saldo < mediana); 
#         ok    (saldo > mediana) 

#.............................................................................

# LEITURA DOS DADOS/ AJUSTES DA BASE

#__________________________________________________________________

ml<- read.csv2('ml.csv', header = TRUE, sep = ';', dec = ',') # lendo os dados

ml$class <- ifelse(ml$saldo < median(ml$saldo), 'risco', 'ok') # definindo a resposta

str(ml)           # estrutura: todas numéricas, menos a resposta (class)
summary(ml)       # muito NA
ncol(ml)          # 48 variáveis
nrow(ml)          # 2394 indivíduos
nrow(na.omit(ml)) # apenas 4 indivíduos com todas as variáveis completas

# selecionando variáveis com menos missing

ml3 <- ml[ ,c(3,4,5,11,12,13,14,15,16,17,18,21,25,29,30
              ,32,33,34,36, 37,38,39,41,48)]

nrow(ml3) - nrow(na.omit(ml3)) # 227 linhas tem NA

ml4 <- na.omit(ml3) # base sem NA
nrow(ml4)           # 2167 indivíduos 

sum(ml4$class == 'risco') # 1046 indivíduos na classe risco
sum(ml4$class == 'ok')    # 1121 indivíduos na classe ok

#.............................................................................

# SEPARANDO A BASE EM TREINO E TESTE

set.seed(19) # semente (permite reprodutibilidade)

indices <- sample(1:nrow(ml4), size = nrow(ml4)*0.7) # selecionando 70% das linhas para treino 

treino <- ml4[indices,] # base de treino

teste <- ml4[-indices,] # base de teste

# para a validação podemos simular uns 2 ou 3 perfis
# e colocar uns municípios reais

#.............................................................................

# MÉTODOS

# 1.  CART (method = 'rpart')
# 2.  Bagging (method = 'treebag')
# 3.  Random Forest (method = 'ranger', 'Rborist', 'rf')
# 4.  Random Forest by Randomization (method = 'extraTrees')
# 5.  Boosting (method = 'adaboost')
# 6.  Support Vector Machine with Polynomial Kernel (method = 'svmPoly')
# 7.  Support Vector Machines with Radial Basis Function Kernel (method = 'svmRadial')
# 8.  Support Vector Machines with Linear Kernel (method = 'svmLinear')
# 9.  k-Nearest Neighbors (method = 'kknn')
# 10. glmnet (method = 'glmnet') 
# 11. glm (method = 'glmn')
# 12. Análise de discriminante linear (method = 'lda')
# 13. Análise de discriminante quadrático (method = 'rda')

#.............................................................................

# MODELOS

#install.packages("caret")
library(caret) # chamando o pacote

# Especificação da validação cruzada nos dados de treino.

val <- trainControl(method = "repeatedcv",
                    number = 5, # five-fold
                    repeats = 5, # repetir o five-fold 5 vezes
                    summaryFunction = twoClassSummary,
                    classProbs = TRUE)

?caret::train() # documentação da função 

# links úteis

# http://topepo.github.io/caret/train-models-by-tag.html. 
# http://topepo.github.io/caret/using-your-own-model-in-train.html

#______________________________________________________________________________________

# 1.  CART (method = 'rpart')

set.seed(1)

m1 <- train(
  class ~ .,
  data = treino,
  method = "rpart",
  metric = "ROC",
  # tuneGrid = ,
  # tuneLength = ,
  trControl = val)

#______________________________________________________________________________________

# 2.  Bagging (method = 'treebag')

set.seed(1)

m2 <- train(
  class ~ .,
  data = treino,
  method = "treebag",
  # tuneGrid = ,
  # tuneLength = ,
  metric = "ROC",
  trControl = val)

#______________________________________________________________________________________

# 3.  Random Forest (method = 'ranger', 'Rborist', 'rf')

set.seed(1)

m3 <- train(
  class ~ .,
  data = treino,
  method = "ranger",
  # tuneGrid = ,
  # tuneLength = ,
  metric = "ROC",
  trControl = val)


#______________________________________________________________________________________

# 4.  Random Forest by Randomization (method = 'extraTrees')

# set.seed(1)

# m4 <- train(
#  class ~ .,
#  data = treino,
#  method = "extraTrees",
#  # tuneGrid = ,
#  # tuneLength = ,
#  metric = "ROC",
#  trControl = val)

#______________________________________________________________________________________

# 5.  Boosting (method = 'adaboost')

set.seed(1)

m5 <- train(
  class ~ .,
  data = treino,
  method = "adaboost",
  # tuneGrid = ,
  # tuneLength = ,
  metric = "ROC",
  trControl = val)

#______________________________________________________________________________________

# 6.  Support Vector Machine with Polynomial Kernel (method = 'svmPoly')

# set.seed(1)

# m6 <- train(
#   class ~ .,
#   data = treino,
#   method = "svmPoly",
#   # tuneGrid = ,
#   tuneLength = 10,
#   metric = "ROC",
#   preProcess = c("center", "scale"),
#   trControl = val)

#______________________________________________________________________________________

# 7.  Support Vector Machines with Radial Basis Function Kernel (method = 'svmRadial')

# set.seed(1)

#m7 <- train(
#   class ~ .,
#   data = treino,
#   method = "svmRadial",
#   # tuneGrid = ,
#   tuneLength = 10,
#   metric = "ROC",
#   preProcess = c("center", "scale"),
#   trControl = val)


#______________________________________________________________________________________

# 8.  Support Vector Machines with Linear Kernel (method = 'svmLinear')

set.seed(1)

m8 <- train(
  class ~ .,
  data = treino,
  method = "svmLinear",
  # tuneGrid = ,
  tuneLength = 10,
  metric = "ROC",
  trControl = val)

#______________________________________________________________________________________

# 9.  k-Nearest Neighbors (method = 'kknn')

set.seed(1)

m9 <- train(
  class ~ .,
  data = treino,
  method = "kknn",
  # tuneGrid = ,
  # tuneLength = ,
  metric = "ROC",
  trControl = val)

#______________________________________________________________________________________

# 10.  glmnet (method = 'glmnet') 

set.seed(1)

m10 <- train(
  class ~ .,
  data = treino,
  method = "glmnet",
  # tuneGrid = ,
  # tuneLength = ,
  metric = "ROC",
  trControl = val)

#______________________________________________________________________________________

# 11. glm (method = 'glm')

set.seed(1)

m11 <- train(
  class ~ .,
  data = treino,
  method = "glm",
  # tuneGrid = ,
  # tuneLength = ,
  metric = "ROC",
  trControl = val)

#______________________________________________________________________________________

# 12. Análise de discriminante linear (method = 'lda')

set.seed(1)

m12 <- train(
  class ~ .,
  data = treino,
  method = "lda",
  # tuneGrid = ,
  # tuneLength = ,
  metric = "ROC",
  trControl = val)

#______________________________________________________________________________________

# 13. Análise de discriminante quadrático (method = 'qda')

set.seed(1)

m13 <- train(
  class ~ .,
  data = treino,
  method = "qda",
  # tuneGrid = ,
  # tuneLength = ,
  metric = "ROC",
  trControl = val)

#______________________________________________________________________________________

# COMPARAÇÃO DOS MODELOS

# Matrizes de confusão

c1  <- confusionMatrix(predict(m1,  teste), teste$class) # CART
c2  <- confusionMatrix(predict(m2,  teste), teste$class) # Bagging
c3  <- confusionMatrix(predict(m3,  teste), teste$class) # Random Forest
c4  <- confusionMatrix(predict(m4,  teste), teste$class) # Random Forest by Randomization
c5  <- confusionMatrix(predict(m5,  teste), teste$class) # Boosting
c6  <- confusionMatrix(predict(m6,  teste), teste$class) # Support Vector Machine with Polynomial Kernel
c7  <- confusionMatrix(predict(m7,  teste), teste$class) # Support Vector Machines with Radial Basis Function Kernel
c8  <- confusionMatrix(predict(m8,  teste), teste$class) # Support Vector Machines with Linear Kernel
c9  <- confusionMatrix(predict(m9,  teste), teste$class) # k-Nearest Neighbors
c10 <- confusionMatrix(predict(m10, teste), teste$class) # glmnet
c11 <- confusionMatrix(predict(m11, teste), teste$class) # glm
c12 <- confusionMatrix(predict(m12, teste), teste$class) # Análise de discriminante linear
c13 <- confusionMatrix(predict(m13, teste), teste$class) # Análise de discriminante quadrático

#________________________________________________________________________________________

# Predições

pred1  <- predict(m1,  teste, type = "prob")  # CART
pred2  <- predict(m2,  teste, type = "prob")  # Bagging
pred3  <- predict(m3,  teste, type = "prob")  # Random Forest
pred4  <- predict(m4,  teste, type = "prob") # Random Forest by Randomization
pred5  <- predict(m5,  teste, type = "prob")  # Boosting
pred6  <- predict(m6,  teste, type = "prob") # Support Vector Machine with Polynomial Kernel
pred7  <- predict(m7,  teste, type = "prob") # Support Vector Machines with Radial Basis Function Kernel
pred8  <- predict(m8,  teste, type = "prob")  # Support Vector Machines with Linear Kernel
pred9  <- predict(m9,  teste, type = "prob")  # k-Nearest Neighbors
pred10 <- predict(m10, teste, type = "prob")  # glmnet
pred11 <- predict(m11, teste, type = "prob")  # glm
pred12 <- predict(m12, teste, type = "prob")  # Análise de discriminante linear
pred13 <- predict(m13, teste, type = "prob")  # Análise de discriminante quadrático

#________________________________________________________________________________________

library(pROC)

# AUC's
roc1  <- round(roc(teste$class, pred1 [, "risco"])$auc, 3)  # CART
roc2  <- round(roc(teste$class, pred2 [, "ok"])$auc, 3)  # Bagging
roc3  <- round(roc(teste$class, pred3 [, "ok"])$auc, 3)  # Random Forest
roc4  <- round(roc(teste$class, pred4 [, "ok"])$auc, 3) # Random Forest by Randomization
roc5  <- round(roc(teste$class, pred5 [, "ok"])$auc, 3)  # Boosting
roc6  <- round(roc(teste$class, pred6 [, "ok"])$auc, 3) # Support Vector Machine with Polynomial Kernel
roc7  <- round(roc(teste$class, pred7 [, "ok"])$auc, 3) # Support Vector Machines with Radial Basis Function Kernel
roc8  <- round(roc(teste$class, pred8 [, "ok"])$auc, 3)  # Support Vector Machines with Linear Kernel
roc9  <- round(roc(teste$class, pred9 [, "ok"])$auc, 3)  # k-Nearest Neighbors
roc10 <- round(roc(teste$class, pred10[, "ok"])$auc, 3)  # glmnet
roc11 <- round(roc(teste$class, pred11[, "ok"])$auc, 3)  # glm
roc12 <- round(roc(teste$class, pred12[, "ok"])$auc, 3)  # Análise de discriminante linear
roc13 <- round(roc(teste$class, pred13[, "ok"])$auc, 3)  # Análise de discriminante quadrático

#________________________________________________________________________________________

# Tabela de classificações corretas e incorretas

tab <- data.frame("Modelo" = c('CART',
                               'Bagging', 
                               'R Forest ',
                               'R. Forest (Rand)',
                               'Boosting',
                               'SVM Polynomial Kernel',
                               'SVM Radial Basis Kernel',
                               'SVMLinear Kernel',
                               'k-Nearest Neighbors',
                               'glmnet',
                               'glm ',
                               'An. disc. linear ',
                               'An. disc. quadrático'),
                  "(ok|ok)"       = rep(NA, 13),
                  "(ok|risco)"    = rep(NA, 13),
                  "(risco|risco)" = rep(NA, 13),
                  "(risco|ok)"    = rep(NA, 13),
                  check.names = FALSE)


# preenchendo a tabela com as quantidades

# 1.  CART (method = 'rpart')

tab[1, 2] <- c1$table[1,1]
tab[1, 3] <- c1$table[1,2]
tab[1, 4] <- c1$table[2,2]
tab[1, 5] <- c1$table[2,1]

# 2.  Bagging (method = 'treebag')

tab[2, 2] <- c2$table[1,1]
tab[2, 3] <- c2$table[1,2]
tab[2, 4] <- c2$table[2,2]
tab[2, 5] <- c2$table[2,1]

# 3.  Random Forest (method = 'ranger', 'Rborist', 'rf')

tab[3, 2] <- c3$table[1,1]
tab[3, 3] <- c3$table[1,2]
tab[3, 4] <- c3$table[2,2]
tab[3, 5] <- c3$table[2,1]

# 4.  Random Forest by Randomization (method = 'extraTrees')

tab[4, 2] <- c4$table[1,1]
tab[4, 3] <- c4$table[1,2]
tab[4, 4] <- c4$table[2,2]
tab[4, 5] <- c4$table[2,1]

# 5.  Boosting (method = 'adaboost')

tab[5, 2] <- c5$table[1,1]
tab[5, 3] <- c5$table[1,2]
tab[5, 4] <- c5$table[2,2]
tab[5, 5] <- c5$table[2,1]

# 6.  Support Vector Machine with Polynomial Kernel (method = 'svmPoly')

tab[6, 2] <- c6$table[1,1]
tab[6, 3] <- c6$table[1,2]
tab[6, 4] <- c6$table[2,2]
tab[6, 5] <- c6$table[2,1]

# 7.  Support Vector Machines with Radial Basis Function Kernel (method = 'svmRadial')

tab[7, 2] <- c7$table[1,1]
tab[7, 3] <- c7$table[1,2]
tab[7, 4] <- c7$table[2,2]
tab[7, 5] <- c7$table[2,1]

# 8.  Support Vector Machines with Linear Kernel (method = 'svmLinear')

tab[8, 2] <- c8$table[1,1]
tab[8, 3] <- c8$table[1,2]
tab[8, 4] <- c8$table[2,2]
tab[8, 5] <- c8$table[2,1]

# 9.  k-Nearest Neighbors (method = 'kknn')

tab[9, 2] <- c9$table[1,1]
tab[9, 3] <- c9$table[1,2]
tab[9, 4] <- c9$table[2,2]
tab[9, 5] <- c9$table[2,1]

# 10. glmnet (method = 'glmnet') 

tab[10, 2] <- c10$table[1,1]
tab[10, 3] <- c10$table[1,2]
tab[10, 4] <- c10$table[2,2]
tab[10, 5] <- c10$table[2,1]

# 11. glm (method = 'glmn')

tab[11, 2] <- c11$table[1,1]
tab[11, 3] <- c11$table[1,2]
tab[11, 4] <- c11$table[2,2]
tab[11, 5] <- c11$table[2,1]

# 12. Análise de discriminante linear (method = 'lda')

tab[12, 2] <- c12$table[1,1]
tab[12, 3] <- c12$table[1,2]
tab[12, 4] <- c12$table[2,2]
tab[12, 5] <- c12$table[2,1]

# 13. Análise de discriminante quadrático (method = 'rda')

tab[13, 2] <- c13$table[1,1]
tab[13, 3] <- c13$table[1,2]
tab[13, 4] <- c13$table[2,2]
tab[13, 5] <- c13$table[2,1]

tab <- na.omit(tab)
tab

library(xtable)
library(knitr)
kable(tab)

#________________________________________________________________________________________

# Tabela de classificações corretas e incorretas

tab3 <- data.frame("Modelo" = c('CART',
                               'Bagging', 
                               'R Forest ',
                               'R. Forest (Rand)',
                               'Boosting',
                               'SVM Polynomial Kernel',
                               'SVM Radial Basis Kernel',
                               'SVMLinear Kernel',
                               'k-Nearest Neighbors',
                               'glmnet',
                               'glm ',
                               'An. disc. linear ',
                               'An. disc. quadrático'),
                  "(ok|ok)"       = rep(NA, 13),
                  "(ok|risco)"    = rep(NA, 13),
                  "(risco|risco)" = rep(NA, 13),
                  "(risco|ok)"    = rep(NA, 13),
                  check.names = FALSE)


# preenchendo a tabela com as quantidades

# 1.  CART (method = 'rpart')

tab3[1, 2] <- round(c1$table[1,1]/nrow(teste), 3)
tab3[1, 3] <- round(c1$table[1,2]/nrow(teste), 3)
tab3[1, 4] <- round(c1$table[2,2]/nrow(teste), 3)
tab3[1, 5] <- round(c1$table[2,1]/nrow(teste), 3)

# 2.  Bagging (method = 'treebag')

tab3[2, 2] <- round(c2$table[1,1]/nrow(teste), 3)
tab3[2, 3] <- round(c2$table[1,2]/nrow(teste), 3)
tab3[2, 4] <- round(c2$table[2,2]/nrow(teste), 3)
tab3[2, 5] <- round(c2$table[2,1]/nrow(teste), 3)

# 3.  Random Forest (method = 'ranger', 'Rborist', 'rf')

tab3[3, 2] <- round(c3$table[1,1]/nrow(teste), 3)
tab3[3, 3] <- round(c3$table[1,2]/nrow(teste), 3)
tab3[3, 4] <- round(c3$table[2,2]/nrow(teste), 3)
tab3[3, 5] <- round(c3$table[2,1]/nrow(teste), 3)

# 4.  Random Forest by Randomization (method = 'extraTrees')

tab3[4, 2] <- round(c4$table[1,1]/nrow(teste), 3)
tab3[4, 3] <- round(c4$table[1,2]/nrow(teste), 3)
tab3[4, 4] <- round(c4$table[2,2]/nrow(teste), 3)
tab3[4, 5] <- round(c4$table[2,1]/nrow(teste), 3)

# 5.  Boosting (method = 'adaboost')

tab3[5, 2] <- round(c5$table[1,1]/nrow(teste), 3)
tab3[5, 3] <- round(c5$table[1,2]/nrow(teste), 3)
tab3[5, 4] <- round(c5$table[2,2]/nrow(teste), 3)
tab3[5, 5] <- round(c5$table[2,1]/nrow(teste), 3)

# 6.  Support Vector Machine with Polynomial Kernel (method = 'svmPoly')

tab3[6, 2] <- round(c6$table[1,1]/nrow(teste), 3)
tab3[6, 3] <- round(c6$table[1,2]/nrow(teste), 3)
tab3[6, 4] <- round(c6$table[2,2]/nrow(teste), 3)
tab3[6, 5] <- round(c6$table[2,1]/nrow(teste), 3)

# 7.  Support Vector Machines with Radial Basis Function Kernel (method = 'svmRadial')

tab3[7, 2] <- round(c7$table[1,1]/nrow(teste), 3)
tab3[7, 3] <- round(c7$table[1,2]/nrow(teste), 3)
tab3[7, 4] <- round(c7$table[2,2]/nrow(teste), 3)
tab3[7, 5] <- round(c7$table[2,1]/nrow(teste), 3)

# 8.  Support Vector Machines with Linear Kernel (method = 'svmLinear')

tab3[8, 2] <- round(c8$table[1,1]/nrow(teste), 3)
tab3[8, 3] <- round(c8$table[1,2]/nrow(teste), 3)
tab3[8, 4] <- round(c8$table[2,2]/nrow(teste), 3)
tab3[8, 5] <- round(c8$table[2,1]/nrow(teste), 3)

# 9.  k-Nearest Neighbors (method = 'kknn')

tab3[9, 2] <- round(c9$table[1,1]/nrow(teste), 3)
tab3[9, 3] <- round(c9$table[1,2]/nrow(teste), 3)
tab3[9, 4] <- round(c9$table[2,2]/nrow(teste), 3)
tab3[9, 5] <- round(c9$table[2,1]/nrow(teste), 3)

# 10. glmnet (method = 'glmnet') 

tab3[10, 2] <- round(c10$table[1,1]/nrow(teste), 3)
tab3[10, 3] <- round(c10$table[1,2]/nrow(teste), 3)
tab3[10, 4] <- round(c10$table[2,2]/nrow(teste), 3)
tab3[10, 5] <- round(c10$table[2,1]/nrow(teste), 3)

# 11. glm (method = 'glmn')

tab3[11, 2] <- round(c11$table[1,1]/nrow(teste), 3)
tab3[11, 3] <- round(c11$table[1,2]/nrow(teste), 3)
tab3[11, 4] <- round(c11$table[2,2]/nrow(teste), 3)
tab3[11, 5] <- round(c11$table[2,1]/nrow(teste), 3)

# 12. Análise de discriminante linear (method = 'lda')

tab3[12, 2] <- round(c12$table[1,1]/nrow(teste), 3)
tab3[12, 3] <- round(c12$table[1,2]/nrow(teste), 3)
tab3[12, 4] <- round(c12$table[2,2]/nrow(teste), 3)
tab3[12, 5] <- round(c12$table[2,1]/nrow(teste), 3)

# 13. Análise de discriminante quadrático (method = 'rda')

tab3[13, 2] <- round(c13$table[1,1]/nrow(teste), 3)
tab3[13, 3] <- round(c13$table[1,2]/nrow(teste), 3)
tab3[13, 4] <- round(c13$table[2,2]/nrow(teste), 3)
tab3[13, 5] <- round(c13$table[2,1]/nrow(teste), 3)

tab3 <- na.omit(tab3)
tab3

library(xtable)
library(knitr)
kable(tab3)

#________________________________________________________________________________________

# tabela de medidas de qualidade preditiva

tab2 <- data.frame("Modelo" = c('CART',
                               'Bagging', 
                               'R Forest ',
                               'R. Forest (Rand)',
                               'Boosting',
                               'SVM Polynomial Kernel',
                               'SVM Radial Basis Kernel',
                               'SVMLinear Kernel',
                               'k-Nearest Neighbors',
                               'glmnet',
                               'glm ',
                               'An. disc. linear ',
                               'An. disc. quadrático'),
                  "sensibilidade"       = rep(NA, 13),
                  "especificidade"    = rep(NA, 13),
                  "acurácia" = rep(NA, 13),
                  "AUC"    = rep(NA, 13),
                  check.names = FALSE)

# preenchendo a tabela com as quantidades

# 1.  CART (method = 'rpart')

tab2[1, 2] <- round(c1$byClass[1], 3)
tab2[1, 3] <- round(c1$byClass[2], 3)
tab2[1, 4] <- round(c1$overall[1],3)
tab2[1, 5] <- roc1

# 2.  Bagging (method = 'treebag')

tab2[2, 2] <- round(c2$byClass[1], 3)
tab2[2, 3] <- round(c2$byClass[2], 3)
tab2[2, 4] <- round(c2$overall[1],3)
tab2[2, 5] <- roc2

# 3.  Random Forest (method = 'ranger', 'Rborist', 'rf')

tab2[3, 2] <- round(c3$byClass[1], 3)
tab2[3, 3] <- round(c3$byClass[2], 3)
tab2[3, 4] <- round(c3$overall[1],3)
tab2[3, 5] <- roc3

# 4.  Random Forest by Randomization (method = 'extraTrees')

tab2[4, 2] <- round(c4$byClass[1], 3)
tab2[4, 3] <- round(c4$byClass[2], 3)
tab2[4, 4] <- round(c4$overall[1],3)
tab2[4, 5] <- roc4

# 5.  Boosting (method = 'adaboost')

tab2[5, 2] <- round(c5$byClass[1], 3)
tab2[5, 3] <- round(c5$byClass[2], 3)
tab2[5, 4] <- round(c5$overall[1],3)
tab2[5, 5] <- roc5

# 6.  Support Vector Machine with Polynomial Kernel (method = 'svmPoly')

tab2[6, 2] <- round(c6$byClass[1], 3)
tab2[6, 3] <- round(c6$byClass[2], 3)
tab2[6, 4] <- round(c6$overall[1],3)
tab2[6, 5] <- roc6

# 7.  Support Vector Machines with Radial Basis Function Kernel (method = 'svmRadial')

tab2[7, 2] <- round(c7$byClass[1], 3)
tab2[7, 3] <- round(c7$byClass[2], 3)
tab2[7, 4] <- round(c7$overall[1],3)
tab2[7, 5] <- roc7

# 8.  Support Vector Machines with Linear Kernel (method = 'svmLinear')

tab2[8, 2] <- round(c8$byClass[1], 3)
tab2[8, 3] <- round(c8$byClass[2], 3)
tab2[8, 4] <- round(c8$overall[1],3)
tab2[8, 5] <- roc8

# 9.  k-Nearest Neighbors (method = 'kknn')

tab2[9, 2] <- round(c9$byClass[1], 3)
tab2[9, 3] <- round(c9$byClass[2], 3)
tab2[9, 4] <- round(c9$overall[1],3)
tab2[9, 5] <- roc9

# 10. glmnet (method = 'glmnet') 

tab2[10, 2] <- round(c10$byClass[1], 3)
tab2[10, 3] <- round(c10$byClass[2], 3)
tab2[10, 4] <- round(c10$overall[1],3)
tab2[10, 5] <- roc10

# 11. glm (method = 'glm')

tab2[11, 2] <- round(c11$byClass[1], 3)
tab2[11, 3] <- round(c11$byClass[2], 3)
tab2[11, 4] <- round(c11$overall[1],3)
tab2[11, 5] <- roc11

# 12. Análise de discriminante linear (method = 'lda')

tab2[12, 2] <- round(c12$byClass[1], 3)
tab2[12, 3] <- round(c12$byClass[2], 3)
tab2[12, 4] <- round(c12$overall[1],3)
tab2[12, 5] <- roc12

# 13. Análise de discriminante quadrático (method = 'rda')

tab2[13, 2] <- round(c13$byClass[1], 3)
tab2[13, 3] <- round(c13$byClass[2], 3)
tab2[13, 4] <- round(c13$overall[1],3)
tab2[13, 5] <- roc13

tab2 <- na.omit(tab2)
tab2

library(xtable)
library(knitr)
kable(tab2)

#________________________________________________________________________________________

# GRÁFICOS

# Curva ROC
plot (roc(teste$class, pred1 [, "risco"]), asp = 1, col = 1, 
      ylab = 'Sensibilidade', xlab = 'Especificidade', xlim = c(1:0), ylim = c(0:1))

lines(roc(teste$class, pred1 [, "risco"]), col = 1)   # CART
lines(roc(teste$class, pred2 [, "ok"]), col = 2)   # Bagging
lines(roc(teste$class, pred3 [, "ok"]), col = 3)   # Random Forest
lines(roc(teste$class, pred4 [, "ok"]), col = 4)   # Random Forest by Randomization
lines(roc(teste$class, pred5 [, "ok"]), col = 4)   # Boosting
lines(roc(teste$class, pred6 [, "ok"]), col = 6)   # Support Vector Machine with Polynomial Kernel
lines(roc(teste$class, pred7 [, "ok"]), col = 7)   # Support Vector Machines with Radial Basis Function Kernel
lines(roc(teste$class, pred8 [, "ok"]), col = 5)   # Support Vector Machines with Linear Kernel
lines(roc(teste$class, pred9 [, "ok"]), col = 6)   # k-Nearest Neighbors
lines(roc(teste$class, pred10[, "ok"]), col = 7)   # glmnet
lines(roc(teste$class, pred11[, "ok"]), col = 8)   # glm
lines(roc(teste$class, pred12[, "ok"]), col = 9)   # Análise de discriminante linear
lines(roc(teste$class, pred13[, "ok"]), col = 10)  # Análise de discriminante quadrático

legend("bottomright", legend=c("cart", "bag.", "rf", "boost.", "svm",
                           "knn", "glmnet", "glm", "lda", "qda"), 
       lty=1, col=c(1:12), lwd=2, bty="n")


barplot(tab2$acurácia, main = 'Acurácia', 
        names.arg = c('m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12'))

barplot(tab2$AUC, main = 'Área sob a curva ROC')

barplot(tab$`(ok|ok)`, main = 'ok|ok')

barplot(tab$`(risco|risco)`, main = 'risco|risco')

barplot(tab$`(ok|risco)`, main = 'ok|risco')

barplot(tab$`(risco|ok)`, main = 'risco|ok')

#________________________________________________________________________________________