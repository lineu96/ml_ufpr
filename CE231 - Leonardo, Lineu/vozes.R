#.............................................................................
# MACHINE LEARNING
#.............................................................................

# Lineu Alberto C. de Freitas - GRR20149144
# Leonardo Henrique B. Kruger - GRR20149101

#.............................................................................
#ENCODING: UTF-8
#.............................................................................

# CONTEXTO

# Dados disponiveis no repositorio do kaggle
# https://www.kaggle.com/primaryobjects/voicegender/data

# Objetivo - classificar, com base em preditoras, se o audio vem de uma pessoa
# do sexo feminino ou masculino 

#.............................................................................

# VARIÁVEIS

# meanfreq - Frequencia Media (kHz)
# sd       - Desvio padrao da Frequencia
# sp.ent   - Entropia Espectral
# sfm      - Planicidade Espectral (Planeza)
# mode     -  Frequencia Modal
# centroid - Centroide de Frequencia
# peakf    - Frequencia de Pico (Amplitude)
# meanfun  - Frequencia Media Fundamental
# minfun   - Frequencia Minima Fundamental
# maxfun   - Frequencia Maxima Fundamental
# meandom  - Frequencia Media Dominante
# mindom   - Frequencia Minima Dominante
# maxdom   - Frequencia Maxima Dominante
# frange   - Amplitude de Frequencia Dominante
# modindx  - Indice de Modulacao

# label    - Sexo(Feminino|Masculina)

#.............................................................................

# LEITURA DOS DADOS/ AJUSTES DA BASE

#__________________________________________________________________


voiceFULL <- read.csv("voice.csv", header=T, sep=",", stringsAsFactors = FALSE)
voiceFULL$label <- as.factor(voice$label)
levels(voiceFULL$label) <- c("Mulher","Homem")
voice <- voiceFULL[,c(-3,-4,-5,-6,-7,-8)]
voice[, -ncol(voice)] <- sapply(voice[, -ncol(voice)],
                                FUN = scale,
                                center = TRUE,
                                scale = TRUE)


#.............................................................................

# SEPARANDO A BASE EM TREINO E TESTE

set.seed(19) # semente (permite reprodutibilidade)

indices <- sample(1:nrow(voice), size = nrow(voice)*0.7) # selecionando 70% das linhas para treino 

treino <- voice[indices,] # base de treino

teste <- voice[-indices,] # base de teste

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
  label ~ .,
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
  label ~ .,
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
  label ~ .,
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
  label ~ .,
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

set.seed(1)

m7 <- train(
  label ~ .,
   data = treino,
   method = "svmRadial",
   # tuneGrid = ,
   tuneLength = 10,
   metric = "ROC",
   preProcess = c("center", "scale"),
   trControl = val)


#______________________________________________________________________________________

# 8.  Support Vector Machines with Linear Kernel (method = 'svmLinear')

set.seed(1)

m8 <- train(
  label ~ .,
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
  label ~ .,
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
  label ~ .,
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
  label ~ .,
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
  label ~ .,
  data = treino,
  method = "lda",
  # tuneGrid = ,
  # tuneLength = ,
  metric = "ROC",
  trControl = val)

#______________________________________________________________________________________

# 13. Análise de discriminante quadrático (method = 'qda')

#set.seed(1)

#m13 <- train(
#  label ~ .,
#  data = treino,
#  method = "qda",
#  # tuneGrid = ,
#  # tuneLength = ,
#  metric = "ROC",
#  trControl = val)

#______________________________________________________________________________________

# COMPARAÇÃO DOS MODELOS

# Matrizes de confusão

c1  <- confusionMatrix(predict(m1,  teste), teste$label) # CART
c2  <- confusionMatrix(predict(m2,  teste), teste$label) # Bagging
c3  <- confusionMatrix(predict(m3,  teste), teste$label) # Random Forest
c4  <- confusionMatrix(predict(m4,  teste), teste$label) # Random Forest by Randomization
c5  <- confusionMatrix(predict(m5,  teste), teste$label) # Boosting
c6  <- confusionMatrix(predict(m6,  teste), teste$label) # Support Vector Machine with Polynomial Kernel
c7  <- confusionMatrix(predict(m7,  teste), teste$label) # Support Vector Machines with Radial Basis Function Kernel
c8  <- confusionMatrix(predict(m8,  teste), teste$label) # Support Vector Machines with Linear Kernel
c9  <- confusionMatrix(predict(m9,  teste), teste$label) # k-Nearest Neighbors
c10 <- confusionMatrix(predict(m10, teste), teste$label) # glmnet
c11 <- confusionMatrix(predict(m11, teste), teste$label) # glm
c12 <- confusionMatrix(predict(m12, teste), teste$label) # Análise de discriminante linear
c13 <- confusionMatrix(predict(m13, teste), teste$label) # Análise de discriminante quadrático

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
roc1  <- round(roc(teste$label, pred1 [, "Mulher"])$auc, 3)  # CART
roc2  <- round(roc(teste$label, pred2 [, "Homem"])$auc, 3)  # Bagging
roc3  <- round(roc(teste$label, pred3 [, "Homem"])$auc, 3)  # Random Forest
roc4  <- round(roc(teste$label, pred4 [, "Homem"])$auc, 3) # Random Forest by Randomization
roc5  <- round(roc(teste$label, pred5 [, "Homem"])$auc, 3)  # Boosting
roc6  <- round(roc(teste$label, pred6 [, "Homem"])$auc, 3) # Support Vector Machine with Polynomial Kernel
roc7  <- round(roc(teste$label, pred7 [, "Homem"])$auc, 3) # Support Vector Machines with Radial Basis Function Kernel
roc8  <- round(roc(teste$label, pred8 [, "Homem"])$auc, 3)  # Support Vector Machines with Linear Kernel
roc9  <- round(roc(teste$label, pred9 [, "Homem"])$auc, 3)  # k-Nearest Neighbors
roc10 <- round(roc(teste$label, pred10[, "Homem"])$auc, 3)  # glmnet
roc11 <- round(roc(teste$label, pred11[, "Homem"])$auc, 3)  # glm
roc12 <- round(roc(teste$label, pred12[, "Homem"])$auc, 3)  # Análise de discriminante linear
roc13 <- round(roc(teste$label, pred13[, "Homem"])$auc, 3)  # Análise de discriminante quadrático

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
                  "(Homem|Homem)"       = rep(NA, 13),
                  "(Homem|Mulher)"    = rep(NA, 13),
                  "(Mulher|Mulher)" = rep(NA, 13),
                  "(Mulher|Homem)"    = rep(NA, 13),
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
                   "(Homem|Homem)"       = rep(NA, 13),
                   "(Homem|Mulher)"    = rep(NA, 13),
                   "(Mulher|Mulher)" = rep(NA, 13),
                   "(Mulher|Homem)"    = rep(NA, 13),
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
plot (roc(teste$label, pred1 [, "Mulher"]), asp = 1, col = 1, 
      ylab = 'Sensibilidade', xlab = 'Especificidade', xlim = c(1:0), ylim = c(0:1))

lines(roc(teste$label, pred1 [, "Mulher"]), col = 1)   # CART
lines(roc(teste$label, pred2 [, "Homem"]), col = 2)   # Bagging
lines(roc(teste$label, pred3 [, "Homem"]), col = 3)   # Random Forest
lines(roc(teste$label, pred4 [, "Homem"]), col = 4)   # Random Forest by Randomization
lines(roc(teste$label, pred5 [, "Homem"]), col = 4)   # Boosting
lines(roc(teste$label, pred6 [, "Homem"]), col = 6)   # Support Vector Machine with Polynomial Kernel
lines(roc(teste$label, pred7 [, "Homem"]), col = 7)   # Support Vector Machines with Radial Basis Function Kernel
lines(roc(teste$label, pred8 [, "Homem"]), col = 5)   # Support Vector Machines with Linear Kernel
lines(roc(teste$label, pred9 [, "Homem"]), col = 6)   # k-Nearest Neighbors
lines(roc(teste$label, pred10[, "Homem"]), col = 7)   # glmnet
lines(roc(teste$label, pred11[, "Homem"]), col = 8)   # glm
lines(roc(teste$label, pred12[, "Homem"]), col = 9)   # Análise de discriminante linear
lines(roc(teste$label, pred13[, "Homem"]), col = 10)  # Análise de discriminante quadrático

legend("bottomright", legend=c('cart',
                                'bag.', 
                               'rf ',
                               'boost',
                               'svm Radial',
                               'svm Linear',
                               'knn',
                               'glmnet',
                               'glm ',
                               'lda'), 
       lty=1, col=c(1:10), lwd=2, bty="n")


barplot(tab2$acurácia, main = 'Acurácia')

barplot(tab2$AUC, main = 'Área sob a curva ROC')

barplot(tab$`(Homem|Homem)`, main = 'Homem|Homem')

barplot(tab$`(Mulher|Mulher)`, main = 'Mulher|Mulher')

barplot(tab$`(Homem|Mulher)`, main = 'Homem|Mulher')

barplot(tab$`(Mulher|Homem)`, main = 'Mulher|Homem')

#________________________________________________________________________________________