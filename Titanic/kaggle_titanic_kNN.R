library(tidyverse)
library(mlr)
train <- read_csv("data/titanic/train.csv")
test <- read_csv("data/titanic/test.csv")
dat <- train %>%
  mutate(Survived = factor(Survived)) %>%
  select(-Name) %>%
  as.data.frame()
  

dat %>%
  ggplot(aes(Pclass,Fare, col=Survived, shape = Survived)) +
  geom_point() +
  theme_bw()
dat %>%
  ggplot(aes(Pclass,Age, col=Survived, shape = Survived)) +
  geom_point() +
  theme_bw()

dat_cv <- dat %>%
  select(Survived:Pclass, Sex, Age, Parch, Parch) %>%
  mutate(Sex = ifelse(Sex=="male", 0, 1))

dat_cv <- impute(dat_cv, cols = list(Age = imputeMean()))
titanicTask <- makeClassifTask(data = dat_cv$data, target = "Survived")
knnParamSpace <- makeParamSet(makeDiscreteParam("k", values = 1:10))
gridSearch <- makeTuneControlGrid()
inner <- makeResampleDesc("CV")
outer <- makeResampleDesc("CV", iters = 10, stratify = TRUE)
knnWrapper <- makeTuneWrapper("classif.knn",
                resampling = inner,
                par.set = knnParamSpace,
                control = gridSearch)

holdoutCVwithTuning <- resample(knnWrapper, 
                                titanicTask,
                                resampling = outer)


kSamples <- map_dbl(1:10, ~resample(  knnWrapper, titanicTask, resampling = outer)$aggr  )
hist(kSamples, xlim = c(0, 0.30))

train()
knn <- makeLearner("classif.knn", par.vals = list("k"=4))
tunedKnnModel <- train(knn, titanicTask)

test_cv <- test %>%
  select(Pclass, Sex, Age, Parch, Parch) %>%
  mutate(Sex = ifelse(Sex=="male", 0, 1))
test_cv <- impute(test_cv, cols = list(Age = imputeMean()))


Survived<- predict(tunedKnnModel, newdata = test_cv$data) %>% as.data.frame()
predFinal <- cbind(test,Survived) %>%
  select(PassengerId, response) %>%
  rename(Survived = response) %>%
  write_csv(path = "data/titanic/submission_02_19_2021.csv")
