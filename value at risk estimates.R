#### LOADING LIBRARIES ####

# Required to run the na.approx command used later for interpolating between known 
# data points.
library(zoo)

# Loading the tseries library, which will help download historical financial
# data straight from the Internet, using the website finance.yahoo.com.
library(tseries)

# Loading the package required for copula modelling.
library(VineCopula)

# Loading the package required for marginal modelling. It includes both AR
# and GARCH models. Also helps to run Jarque--Bera test for normality.
library(fGarch)

# Loading the package with the Kolmogorov-Smirnov test, which tests for
# uniformity. 
library(KScorrect)

# Loading the statistics package to help with testing for normality.
# jarqueberaTest is the command used for this.
library(stats)

# Loading the package with the Anderson-Darling test, another test for 
# uniformity. Will help in checking if samples look like 
# those from a standard uniform distribution.
library(ADGofTest)



#### LOADING & PROCESSING DATA ####

# Obtaining weekly prices of S&P500 from 01-02-2010 to 20-01-2020 from finance.yahoo.com 
SandPprices = get.hist.quote(instrument = "^gspc", start = "2010-02-01", end = "2020-01-20", compression = "w", quote="AdjClose")

# Obtaining weekly prices of SSE Composite Index from 01-02-2010 to 13-01-2020 from finance.yahoo.com 
SSEprices = get.hist.quote(instrument = "000001.SS", start = "2010-02-01", end = "2020-01-13", compression = "w", quote="AdjClose")


# Checking for and counting how many missing weekly prices are in the 
# S&P500 data. 0 missing values are found.
sum(is.na(SandPprices))

# Checking for and counting how many missing weekly prices are in the
# SSE Composite Index data. 10 missing values are found.
sum(is.na(SSEprices))

# Fill in missing values in SSE weekly prices data by using na.approx command
# to interpolate between known values.
SSEprices <- na.approx(SSEprices)


# Convert the S&P500 prices into log-returns 
SandPret = diff(log(SandPprices))

# Convert the SSE prices into log-returns
SSEret = diff(log(SSEprices))


# Plotting the S&P500 log-returns with time
plot(SandPret, xlab = "Year", ylab = "log-returns")

# Plotting the SSE log-returns with time
plot(SSEret, xlab = "Year", ylab = "log-returns")


# Extracting dates from returns data for S&P500
SandP_ret = coredata(SandPret) 

# Renaming column 'Adjusted' to 'S&P500' in the S&P500 log-returns data.
colnames(SandP_ret) <- c("S&P500")


# Extracting dates from returns data for SSE Composite Index
SSE_ret = coredata(SSEret)

# Renaming column 'Adjusted' to 'SSE' in the SSE Composite Index 
# log-returns data.
colnames(SSE_ret) <- c("SSE")


# Testing for normality in the log-returns data for S&P500 using the 
# Jarque--Bera test. Low p-value of less than 2.2e-16 is strong evidence against the
# null hypothesis, which is that the data comes from a normal
# distribution. Therefore, this null hypothesis is rejected.
jarqueberaTest(SandP_ret)

# Testing for normality in the log-returns data for SSE Composite Index
# using the Jarque--Bera test. Low p-value of less than 2.2e-16 is strong evidence 
# against the null hypothesis, which is that the data comes from a normal
# distribution. Therefore, this null hypothesis is rejected.
jarqueberaTest(SSE_ret)



#### Building AR models using Box-Jenkins approach ####

# IDENTIFICATION STEP #

# S&P500 log-returns

par(mfrow=c(2,2), bg = "lightgreen")

# The ACF (auto-correlation function) plot shows the correlation between 
# Yt and Yt-k, that are separated by k time periods.
acf(SandP_ret, col="blue", lwd=2)

# The PACF (partial auto-correlation function) plot shows the correlation 
# between Yt and Yt-k once the effects in between (e.g Yt-k+1, Yt-k+2 etc.) are 
# removed. This plot also helps to identify the order of the AR model to be used. 
# In this plot, the most significant lag spike can be observed at lag 1.
pacf(SandP_ret, col="blue", lwd=2)

# This ACF plot indicates that if there is auto-correlation present within
# squared log-returns, then ARCH/GARCH effects are present - conditional 
# variances are auto-correlated and depend on previous values of log-returns.
acf(SandP_ret^2, col="red", lwd=2)

par(mfrow=c(1,1))


# SSE Composite Index log-returns

par(mfrow=c(2,2), bg = "lightgreen")

# The ACF plot shows the correlation between Yt and Yt-k, that are
# separated by k time periods.
acf(SSE_ret, col="blue", lwd=2)

# The PACF plot shows the correlation between Yt and Yt-k once the 
# effects in between (e.g Yt-k+1, Yt-k+2 etc.) are removed. This plot also helps 
# to identify the order of the AR model to be used. In this plot,
# the most significant lag spike can be observed at lag 1. 
pacf(SSE_ret, col="blue", lwd=2)

# This ACF plot indicates that if there is auto-correlation present within
# squared log-returns, then ARCH/GARCH effects are present - conditional 
# variances are auto-correlated and depend on previous values of log-returns.
acf(SSE_ret^2, col="red", lwd=2)
par(mfrow=c(1,1))


# ESTIMATION STEP #

## MODEL 1 ##
# Below, I test 9 different combinations of GARCH parameters and change the conditional
# distribution between the normal, student-t and skewed student-t distributions in order
# to find the model of best fit. The AR order is set to 1, as the PACF plot for
# S&P500 above indicates the most significant lag is at lag 1. 
model1_N1=garchFit(formula=~arma(1,0)+garch(1,1),data=SandP_ret,trace=F,cond.dist="norm")
model1_N2=garchFit(formula=~arma(1,0)+garch(1,2),data=SandP_ret,trace=F,cond.dist="norm")
model1_N3=garchFit(formula=~arma(1,0)+garch(1,3),data=SandP_ret,trace=F,cond.dist="norm")
model1_N4=garchFit(formula=~arma(1,0)+garch(2,1),data=SandP_ret,trace=F,cond.dist="norm")
model1_N5=garchFit(formula=~arma(1,0)+garch(2,2),data=SandP_ret,trace=F,cond.dist="norm")
model1_N6=garchFit(formula=~arma(1,0)+garch(2,3),data=SandP_ret,trace=F,cond.dist="norm")
model1_N7=garchFit(formula=~arma(1,0)+garch(3,1),data=SandP_ret,trace=F,cond.dist="norm")
model1_N8=garchFit(formula=~arma(1,0)+garch(3,2),data=SandP_ret,trace=F,cond.dist="norm")
model1_N9=garchFit(formula=~arma(1,0)+garch(3,3),data=SandP_ret,trace=F,cond.dist="norm")

ic1_N =rbind(model1_N1@fit$ics,model1_N2@fit$ics,model1_N3@fit$ics,model1_N4@fit$ics, 
         model1_N5@fit$ics, model1_N6@fit$ics, model1_N7@fit$ics, model1_N8@fit$ics, 
         model1_N9@fit$ics)
rownames(ic1_N) = c("G(1,1)", "G(1,2)", "G(1,3)", "G(2,1)", "G(2,2)", "G(2,3)", 
                 "G(3,1)", "G(3,2)", "G(3,3)")
ic1_N


model1_ST1=garchFit(formula=~arma(1,0)+garch(1,1),data=SandP_ret,trace=F,cond.dist="std")
model1_ST2=garchFit(formula=~arma(1,0)+garch(1,2),data=SandP_ret,trace=F,cond.dist="std")
model1_ST3=garchFit(formula=~arma(1,0)+garch(1,3),data=SandP_ret,trace=F,cond.dist="std")
model1_ST4=garchFit(formula=~arma(1,0)+garch(2,1),data=SandP_ret,trace=F,cond.dist="std")
model1_ST5=garchFit(formula=~arma(1,0)+garch(2,2),data=SandP_ret,trace=F,cond.dist="std")
model1_ST6=garchFit(formula=~arma(1,0)+garch(2,3),data=SandP_ret,trace=F,cond.dist="std")
model1_ST7=garchFit(formula=~arma(1,0)+garch(3,1),data=SandP_ret,trace=F,cond.dist="std")
model1_ST8=garchFit(formula=~arma(1,0)+garch(3,2),data=SandP_ret,trace=F,cond.dist="std")
model1_ST9=garchFit(formula=~arma(1,0)+garch(3,3),data=SandP_ret,trace=F,cond.dist="std")

ic1_ST =rbind(model1_ST1@fit$ics,model1_ST2@fit$ics,model1_ST3@fit$ics,model1_ST4@fit$ics, 
             model1_ST5@fit$ics, model1_ST6@fit$ics, model1_ST7@fit$ics, model1_ST8@fit$ics, 
             model1_ST9@fit$ics)
rownames(ic1_ST) = c("G(1,1)", "G(1,2)", "G(1,3)", "G(2,1)", "G(2,2)", "G(2,3)", 
                    "G(3,1)", "G(3,2)", "G(3,3)")
ic1_ST


model1_SST1=garchFit(formula=~arma(1,0)+garch(1,1),data=SandP_ret,trace=F,cond.dist="sstd")
model1_SST2=garchFit(formula=~arma(1,0)+garch(1,2),data=SandP_ret,trace=F,cond.dist="sstd")
model1_SST3=garchFit(formula=~arma(1,0)+garch(1,3),data=SandP_ret,trace=F,cond.dist="sstd")
model1_SST4=garchFit(formula=~arma(1,0)+garch(2,1),data=SandP_ret,trace=F,cond.dist="sstd")
model1_SST5=garchFit(formula=~arma(1,0)+garch(2,2),data=SandP_ret,trace=F,cond.dist="sstd")
model1_SST6=garchFit(formula=~arma(1,0)+garch(2,3),data=SandP_ret,trace=F,cond.dist="sstd")
model1_SST7=garchFit(formula=~arma(1,0)+garch(3,1),data=SandP_ret,trace=F,cond.dist="sstd")
model1_SST8=garchFit(formula=~arma(1,0)+garch(3,2),data=SandP_ret,trace=F,cond.dist="sstd")
model1_SST9=garchFit(formula=~arma(1,0)+garch(3,3),data=SandP_ret,trace=F,cond.dist="sstd")

ic1_SST =rbind(model1_SST1@fit$ics,model1_SST2@fit$ics,model1_SST3@fit$ics,model1_SST4@fit$ics, 
              model1_SST5@fit$ics, model1_SST6@fit$ics, model1_SST7@fit$ics, model1_SST8@fit$ics, 
              model1_SST9@fit$ics)
rownames(ic1_SST) = c("G(1,1)", "G(1,2)", "G(1,3)", "G(2,1)", "G(2,2)", "G(2,3)", 
                    "G(3,1)", "G(3,2)", "G(3,3)")
ic1_SST

# A 1st order AR model (AR(1)) is chosen, as the most significant lag on partial ACF
# plot occurs at lag 1. Lags above lag 10 are ignored because realistically, events
# that occurred 10 weeks ago should not affect current log-returns. A GARCH(1,1) model
# will be used as there is auto-correlation in squared log-returns to account for,
# as shown with multiple significant spikes in the ACF plot for S&P500's squared 
# log-returns. The student-t distribution is chosen as the conditional distribution
# because even though the skewed student-t distribution models had lower AIC values
# in comparison, they failed to pass all tests for uniformity and auto-correlation.
# The AR(1)-GARCH(1,1) model with student-t distribution had a lower AIC value than 
# all the normal distribution models, and passed all 4 tests, so I chose this model.
# The model cannot be tested with GARCH parameter combinations beyond
# GARCH(3,3), as this will be time consuming.


## MODEL 2 ##
# Below, I test 9 different combinations of GARCH parameters and change the conditional
# distribution between the normal, student-t and skewed student-t distributions in order
# to find the model of best fit. The AR order is set to 1, as the PACF plot for
# SSE Composite Index above indicates that the most significant lag is at lag 1. 
model2_N1=garchFit(formula=~arma(1,0)+garch(1,1),data=SSE_ret,trace=F,cond.dist="norm")
model2_N2=garchFit(formula=~arma(1,0)+garch(1,2),data=SSE_ret,trace=F,cond.dist="norm")
model2_N3=garchFit(formula=~arma(1,0)+garch(1,3),data=SSE_ret,trace=F,cond.dist="norm")
model2_N4=garchFit(formula=~arma(1,0)+garch(2,1),data=SSE_ret,trace=F,cond.dist="norm")
model2_N5=garchFit(formula=~arma(1,0)+garch(2,2),data=SSE_ret,trace=F,cond.dist="norm")
model2_N6=garchFit(formula=~arma(1,0)+garch(2,3),data=SSE_ret,trace=F,cond.dist="norm")
model2_N7=garchFit(formula=~arma(1,0)+garch(3,1),data=SSE_ret,trace=F,cond.dist="norm")
model2_N8=garchFit(formula=~arma(1,0)+garch(3,2),data=SSE_ret,trace=F,cond.dist="norm")
model2_N9=garchFit(formula=~arma(1,0)+garch(3,3),data=SSE_ret,trace=F,cond.dist="norm")

ic2_N =rbind(model2_N1@fit$ics,model2_N2@fit$ics,model2_N3@fit$ics,model2_N4@fit$ics, 
             model2_N5@fit$ics, model2_N6@fit$ics, model2_N7@fit$ics, model2_N8@fit$ics, 
             model2_N9@fit$ics)
rownames(ic2_N) = c("G(1,1)", "G(1,2)", "G(1,3)", "G(2,1)", "G(2,2)", "G(2,3)", 
                    "G(3,1)", "G(3,2)", "G(3,3)")
ic2_N


model2_ST1=garchFit(formula=~arma(1,0)+garch(1,1),data=SSE_ret,trace=F,cond.dist="std")
model2_ST2=garchFit(formula=~arma(1,0)+garch(1,2),data=SSE_ret,trace=F,cond.dist="std")
model2_ST3=garchFit(formula=~arma(1,0)+garch(1,3),data=SSE_ret,trace=F,cond.dist="std")
model2_ST4=garchFit(formula=~arma(1,0)+garch(2,1),data=SSE_ret,trace=F,cond.dist="std")
model2_ST5=garchFit(formula=~arma(1,0)+garch(2,2),data=SSE_ret,trace=F,cond.dist="std")
model2_ST6=garchFit(formula=~arma(1,0)+garch(2,3),data=SSE_ret,trace=F,cond.dist="std")
model2_ST7=garchFit(formula=~arma(1,0)+garch(3,1),data=SSE_ret,trace=F,cond.dist="std")
model2_ST8=garchFit(formula=~arma(1,0)+garch(3,2),data=SSE_ret,trace=F,cond.dist="std")
model2_ST9=garchFit(formula=~arma(1,0)+garch(3,3),data=SSE_ret,trace=F,cond.dist="std")

ic2_ST =rbind(model2_ST1@fit$ics,model2_ST2@fit$ics,model2_ST3@fit$ics,model2_ST4@fit$ics, 
              model2_ST5@fit$ics, model2_ST6@fit$ics, model2_ST7@fit$ics, model2_ST8@fit$ics, 
              model2_ST9@fit$ics)
rownames(ic2_ST) = c("G(1,1)", "G(1,2)", "G(1,3)", "G(2,1)", "G(2,2)", "G(2,3)", 
                     "G(3,1)", "G(3,2)", "G(3,3)")
ic2_ST


model2_SST1=garchFit(formula=~arma(1,0)+garch(1,1),data=SSE_ret,trace=F,cond.dist="sstd")
model2_SST2=garchFit(formula=~arma(1,0)+garch(1,2),data=SSE_ret,trace=F,cond.dist="sstd")
model2_SST3=garchFit(formula=~arma(1,0)+garch(1,3),data=SSE_ret,trace=F,cond.dist="sstd")
model2_SST4=garchFit(formula=~arma(1,0)+garch(2,1),data=SSE_ret,trace=F,cond.dist="sstd")
model2_SST5=garchFit(formula=~arma(1,0)+garch(2,2),data=SSE_ret,trace=F,cond.dist="sstd")
model2_SST6=garchFit(formula=~arma(1,0)+garch(2,3),data=SSE_ret,trace=F,cond.dist="sstd")
model2_SST7=garchFit(formula=~arma(1,0)+garch(3,1),data=SSE_ret,trace=F,cond.dist="sstd")
model2_SST8=garchFit(formula=~arma(1,0)+garch(3,2),data=SSE_ret,trace=F,cond.dist="sstd")
model2_SST9=garchFit(formula=~arma(1,0)+garch(3,3),data=SSE_ret,trace=F,cond.dist="sstd")

ic2_SST =rbind(model2_SST1@fit$ics,model2_SST2@fit$ics,model2_SST3@fit$ics,model2_SST4@fit$ics, 
               model2_SST5@fit$ics, model2_SST6@fit$ics, model2_SST7@fit$ics, model2_SST8@fit$ics, 
               model2_SST9@fit$ics)
rownames(ic2_SST) = c("G(1,1)", "G(1,2)", "G(1,3)", "G(2,1)", "G(2,2)", "G(2,3)", 
                      "G(3,1)", "G(3,2)", "G(3,3)")
ic2_SST

# A 1st order AR model (AR(1)) is chosen, as the most significant lag on partial ACF
# plot occurs at lag 1. Lags above lag 10 are ignored because realistically, events
# that occurred 10 weeks ago should not affect current log-returns. A GARCH(1,1) model
# will be used as there is auto-correlation in squared log-returns to account for,
# as shown with multiple significant spikes in the ACF plot for SSE's Composite Index 
# squared log-returns. The student-t distribution is chosen as the conditional distribution
# with AR(1)-GARCH(1,1), as this model had the lowest AIC value and passed all tests 
# for uniformity and auto-correlation. This confirms that the chosen model is suitable.
# The model cannot be tested with GARCH parameter combinations beyond
# GARCH(3,3), as this will be time consuming.


# MODEL CHECKING STEP #


## S&P500 returns

# Residuals are extracted from the S&P500 model, and standardised in this step.
# These residuals are standardised through dividing them by estimated values
# of sigma-t to give estimates of epsilon-t.
resSandP <- residuals(model1_ST1, standardize=TRUE)
par(mfrow=c(2,1))
# This gives the auto-correlation function plot for the standardised residuals. There
# is no evidence of auto-correlation - there are some very minor spikes but these
# can be a result of sample variation. This shows that the conditional mean has been modelled
# well, as there is not much auto-correlation seen in this plot.
acf(resSandP, col="blue", lwd=2)
# This gives the auto-correlation function plot for the square of the standardised
# residuals. There are also no significant lags in this plot, showing that
# the GARCH effects have been modelled reasonably well.
acf(resSandP^2, col="red", lwd=2)
par(mfrow=c(1,1))
# The Ljung-Box test for auto-correlation is carried out to see if there is no
# autocorrelation in the residuals. A high p-value of 0.6727 suggests there is no
# evidence against the null hypothesis of no auto-correlation in the standardised 
# residuals. Therefore, the null hypothesis is not rejected.
Box.test(resSandP, lag = 10, type = c("Ljung-Box"), fitdf = 1)
# The Ljung-Box test for auto-correlation is carried out to see if there is no 
# autocorrelation in the squared standardised residuals. A high p-value of 0.3758 
# suggests there is no evidence against the null hypothesis of no auto-correlation in 
# the squared standardised residuals. Therefore, the null hypothesis is not rejected.
Box.test(resSandP^2, lag = 10, type = c("Ljung-Box"), fitdf = 1)
# Applying the Probability Integral Transform (PIT) in this step. After choosing
# the conditional student-t distribution, the cumulative distribution 
# function of the standardised residuals can be applied, as they are independent and identically
# distributed (i.i.d.).
u1<-pstd(resSandP, mean=0, sd=1)[4:length(SandP_ret)]
# When you take a random variable of any distribution and apply the PIT to it using 
# its own C.D.F., the transformed variable should be a standard uniformly 
# distributed random variable. By plotting the histogram below, 
# uniformity can be observed, but the bar heights still vary. However, I proceed because the Box-Ljung
# tests, as well as the Kolmogorov-Smirnov and Anderson-Darling tests (below), give p-values
# above 0.05.
hist(u1)

# Further distributional checks

# Kolmogorov-Smirnov test
KStest1<-LcKS(u1, cdf = "punif")
KStest1$p.value

# Anderson-Darling test
ADtest1<-ad.test(u1, null="punif")
ADtest1$p.value


# SSE returns

# Residuals from the SSE model are extracted and standardised in this step.
# These residuals are standardised through dividing them by estimated values
# of sigma-t to give estimates of epsilon-t.
resSSE <- residuals(model2_ST1, standardize=TRUE)
par(mfrow=c(2,1))
# This gives the auto-correlation function plot for the standardised residuals. There
# is no evidence of auto-correlation - there are some very minor spikes but these
# can be a result of sample variation. This shows that the conditional mean
# has been modelled well, as there is not much auto-correlation seen in this plot.
acf(resSSE, col="blue", lwd=2)
# This gives the auto-correlation function plot for the square of the standardised
# residuals. There are also no significant lags in this plot, showing that
# the GARCH effects have been modelled reasonably well.
acf(resSSE^2, col="red", lwd=2)
par(mfrow=c(1,1))
# The Ljung-Box test for auto-correlation is carried out to see if there is no
# autocorrelation in the residuals. A p-value of 0.2077 suggests there is no
# evidence against the null hypothesis of no auto-correlation in the standardised 
# residuals. Therefore, the null hypothesis is not rejected.
Box.test(resSSE, lag = 10, type = c("Ljung-Box"), fitdf = 1)
# The Ljung-Box test for auto-correlation is carried out to see if there is no 
# autocorrelation in the squared standardised residuals. A high p-value of 0.9373 
# suggests there is no evidence against the null hypothesis of no auto-correlation in 
# the squared standardised residuals. Therefore, the null hypothesis is not rejected.
Box.test(resSSE^2, lag = 10, type = c("Ljung-Box"), fitdf = 1)
# Applying the Probability Integral Transform (PIT) in this step. After choosing
# the conditional student-t distribution, the cumulative distribution 
# function of the standardised residuals can be applied, as they are independent and identically
# distributed (i.i.d.).
u2<-pstd(resSSE, mean=0, sd=1)[4:length(SSE_ret)]
# When you take a random variable of any distribution and apply the PIT to it using 
# its own C.D.F., the transformed variable should be a standard uniformly 
# distributed random variable. By plotting the histogram below,
# uniformity is shown, but the bar heights still vary. However, I proceed because the Box-Ljung
# tests, as well as the Kolmogorov-Smirnov and Anderson-Darling tests (below), give p-values
# above 0.05.
hist(u2)


# Further distributional checks
# Kolmogorov-Smirnov test
KStest1<-LcKS(u2, cdf = "punif")
KStest1$p.value

# Anderson-Darling test
ADtest1<-ad.test(u2, null="punif")
ADtest1$p.value

#### COPULA MODELLING ####

# The BiCopSelect function is used to fit all different copulas against the data and calculate 
# their respective AIC values. Once AIC is found using the transformed residuals (u1 and u2),
# the model which gives the lowest value of AIC for the copula will be chosen. In this case, the log return data is
# best described by the Rotated Tawn type 2 180 degrees copula.
modelf=BiCopSelect(u1, u2, familyset=NA, selectioncrit="AIC", indeptest=TRUE, level=0.05,se = TRUE)
modelf

# The length of the bivariate sample to be simulated is chosen
N=10000
# Using the BiCopSim command to simulate from the chosen copula above. modelf$par and model$fpar2 are used to 
# source the exact parameter values into this command.
u_sim=BiCopSim(N, family=modelf$family, modelf$par,  modelf$par2)
# Now apply the Inverse Probability Integral Transform to resSandP and resSSE.
resSandP_sim=qstd(u_sim[,1], mean = 0, sd = 1) 
resSSE_sim=qstd(u_sim[,2], mean = 0, sd = 1)

# Obtaining non-standardised residuals for each of the stock indices' log-returns.
resSandP_US <- residuals(model1_ST1, standardize=FALSE)
resSSE_US <- residuals(model2_ST1, standardize=FALSE)

## MODEL 1 ##
# Coef function used to observe coefficient values within model 1. Coefficient values are assigned as follows:
coef(model1_ST1) 
c_hat1 <- coef(model1_ST1)[1]
alpha1 <- coef(model1_ST1)[4]
beta1 <- coef(model1_ST1) [5]
omega1 <- coef(model1_ST1)[3]
ar1 <- coef(model1_ST1)[2]
# Finding last value in the S&P500 log-return dataset:
SandP_ret[519]
# Finding last value in S&P500 residuals dataset:
resSandP_US[519]
# Finding last value of conditional variance directly from the model: 
model1_ST1@h.t[519]

# Calculating value of sigma_1 (associated with S&P500)
sigmasq_1 <- omega1 + (alpha1*(resSandP_US[519]^2)) + (beta1*model1_ST1@h.t[519])
sigma_1 <- sqrt(sigmasq_1)                                                     

# Reintroducing GARCH effects and autocorrelation into S&P500 data using calculated sigma_1 value and coefficients
# from model 1.
SandP_sim <- sigma_1*resSandP_sim + c_hat1 + (ar1*SandP_ret[519])   

## MODEL 2 ##
# Coef function used to observe coefficient values within model 2. Coefficient values are assigned as follows:
coef(model2_ST1)
c_hat2 <- coef(model2_ST1)[1]
alpha2 <- coef(model2_ST1)[4]
beta2 <- coef(model2_ST1)[5]
omega2 <- coef(model2_ST1)[3]
ar2 <- coef(model2_ST1)[2]
# Finding last value in the SSE log-return dataset:
SSE_ret[519]
# Finding last value in SSE residuals dataset:
resSSE_US[519]
# Finding last value of conditional variance directly from the model: 
model2_ST1@h.t[519]
ar2 <- coef(model2_ST1)[2]

# Calculating value of sigma_2 (associated with SSE)
sigmasq_2 <- omega2 + (alpha2*(resSSE_US[519]^2)) + (beta2*model2_ST1@h.t[519])
sigma_2 <- sqrt(sigmasq_2)  

# Reintroducing GARCH effects and autocorrelation into SSE data using calculated sigma_2 value and coefficients
# from model 2.
SSE_sim <- sigma_2*resSSE_sim + c_hat2 + (ar2*SSE_ret[519]) 


# Calculating portfolio log-returns
P_sim <- matrix(0, nrow = N, ncol = 1)
VaR_sim <- matrix(0, nrow = 1, ncol = 2)

# In this step, the log-returns are changed back to simple net returns. Afterwards, I construct the 
# portfolio of simple net returns, followed by forming the equally-weighted portfolio of log-returns.
P_sim=log(1+((exp(SandP_sim)-1)+(exp(SSE_sim)-1))*(1/2))

# The final 99% and 95% Value-at-Risk estimates are as follows:
VaR_sim=quantile(P_sim,c(0.01,0.05))
VaR_sim


