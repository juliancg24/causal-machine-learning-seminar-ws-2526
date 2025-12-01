# Extended Notes: Double / Debiased Machine Learning Talk

These notes back each slide with more details and references to the Chernozhukov et al. (2018) paper *“Double / Debiased Machine Learning for Treatment and Structural Parameters”* and Victor Chernozhukov’s talk.

---

## Slide 2 — The problem: causal effects with many covariates

- The paper’s intro sets up the problem as inference on a low-dimensional parameter \(\theta_0\) in the presence of high-dimensional or highly complex nuisance parameters \(\eta_0\), estimated via ML. This is explicitly stated in the abstract and Section 1.:contentReference[oaicite:17]{index=17}
- The point that ML is good for prediction but not automatically for causal parameters is stressed in the talk, where Chernozhukov emphasizes that good prediction does not guarantee good estimation of a causal parameter and can even be misleading.:contentReference[oaicite:18]{index=18}

---

## Slide 3 — Prediction vs. causal estimation

- The general moment-condition language \(\mathbb{E}[\psi(W;\theta_0,\eta_0)]=0\) is introduced in Section 2 (Moment condition / estimating equations framework).:contentReference[oaicite:19]{index=19}
- The distinction between prediction and causal estimation is illustrated in the Introduction: ML methods are used to estimate complex nuisances, but naive plug-in leads to biased estimators of \(\theta_0\).:contentReference[oaicite:20]{index=20}
- The talk transcript also highlights two main points: (i) ML can predict very well, (ii) naive use produces poor estimators for causal parameters.:contentReference[oaicite:21]{index=21}

---

## Slide 4 — Why econometricians love moment conditions

- The paper uses a GMM-style viewpoint: target parameters are defined by population moment conditions, then estimated by solving empirical analogues. This is formalized in Section 2.1.:contentReference[oaicite:22]{index=22}
- Examples like OLS, IV, and GMM are standard; the paper also references semi-parametric literature where efficient scores come from such moment conditions (e.g.\ Chamberlain 1987, Newey 1994).:contentReference[oaicite:23]{index=23}

---

## Slide 5 — Moment conditions in the partially linear model

- The PLR model is introduced as Example 1.1 in Section 1:  
  \[
  Y = D\theta_0 + g_0(X) + U, \quad \mathbb{E}[U\mid X,D]=0,
  \]
  \[
  D = m_0(X)+V, \quad \mathbb{E}[V\mid X]=0.
  \]:contentReference[oaicite:24]{index=24}
- The naive regression-adjustment moment
  \(\mathbb{E}[(Y - D\theta_0 - g_0(X))D]=0\) corresponds to treating \(g_0\) as known and regressing \(Y-g_0(X)\) on \(D\).
- The propensity-style moment  
  \(\mathbb{E}[(Y - D\theta_0)(D - m_0(X))]=0\) is related to reweighting based on deviations from the propensity score. This is less central in the paper but natural in the PLR setting.
- The orthogonal score  
  \(\psi(W;\theta,g,m)=(Y-\theta D-g(X))(D-m(X))\) is explicitly given in Section 4.1 as a key example of an orthogonal moment.:contentReference[oaicite:25]{index=25}

---

## Slide 7 — Naive regression adjustment: bias decomposition

- The plug-in estimator and its bias decomposition are spelled out in the Introduction. The paper considers sample splitting: estimating \(g_0\) on an auxiliary sample and then forming
  \[
  \hat\theta_0 =
    \Big(\frac{1}{n}\sum D_i^2\Big)^{-1}
    \Big(\frac{1}{n}\sum D_i (Y_i - \hat g_0(X_i))\Big).
  \]:contentReference[oaicite:26]{index=26}
- The scaled estimation error decomposes as
  \[
  \sqrt{n}(\hat\theta_0 - \theta_0) = a+b,
  \]
  where
  \[
  a = (E[D^2])^{-1}\frac{1}{\sqrt{n}}\sum D_i U_i,
  \]
  \[
  b \approx (E[D^2])^{-1}\frac{1}{\sqrt{n}}\sum m_0(X_i)\{g_0(X_i) - \hat g_0(X_i)\}.
  \]:contentReference[oaicite:27]{index=27}
- The paper notes that ML methods yield rates like \(n^{-\varphi_g}\) with \(\varphi_g < 1/2\), implying that the term \(b\) diverges in general at \(\sqrt{n}\)-scale, so the estimator fails to be \(\sqrt{n}\)-consistent. This is discussed under “Regularization bias”.:contentReference[oaicite:28]{index=28}

---

## Slide 8 — Orthogonal (residual) score: bias decomposition

- The orthogonalized construction replaces \(D\) by its residual \(V = D - m_0(X)\), estimated as \(\hat V=D-\hat m(X)\), and defines a DML estimator
  \[
  \check\theta_0 = 
  \Big(\frac{1}{n}\sum \hat V_i D_i\Big)^{-1}
  \Big(\frac{1}{n}\sum \hat V_i (Y_i - \hat g(X_i))\Big),
  \]
  which can be rearranged into the residual regression form used in the slides.:contentReference[oaicite:29]{index=29}
- The decomposition
  \[
  \sqrt{n}(\check\theta_0 - \theta_0) = a^* + b^* + c^*
  \]
  is sketched in the paper:  
  - \(a^* = (E[V^2])^{-1}n^{-1/2}\sum V_iU_i\)  
  - \(b^* = (E[V^2])^{-1}n^{-1/2}\sum (\hat m(X_i) - m_0(X_i))(\hat g(X_i) - g_0(X_i))\)  
  - \(c^*\) is the remainder, controlled via sample splitting.
- The key is that \(b^*\) depends on a product of estimation errors and is of order \(\sqrt{n}\,n^{-(\varphi_m+\varphi_g)}\), which can vanish for \(\varphi_m+\varphi_g>1/2\).:contentReference[oaicite:30]{index=30}

---

## Slide 9 — Neyman orthogonality: definition & PLR example

- Definition 2.1 gives Neyman orthogonality formally: the Gateaux derivative of the moment w.r.t.\ the nuisance at \((\theta_0,\eta_0)\) is zero for all directions in a realization set.:contentReference[oaicite:31]{index=31}
- The notation in the paper is
  \[
  \partial_\eta \mathbb{E}_P[\psi(W;\theta_0,\eta_0)][\eta-\eta_0] = 0.
  \]
- For the PLR score
  \(\psi(W;\theta,g,m) = (Y-\theta D-g(X))(D-m(X))\),
  the paper notes that this orthogonality condition holds w.r.t.\ \(\eta=(g,m)\), which accounts for the improved robustness (the derivative of the moment w.r.t.\ misspecification in \(g,m\) vanishes).:contentReference[oaicite:32]{index=32}

---

## Slide 10 — Residual interpretation & IV-style view

- The residual interpretation
  \(\tilde Y = Y-\hat g(X)\), \(\tilde D = D-\hat m(X)\)  
  and the estimator
  \[
  \hat\theta = \frac{\sum \tilde D_i \tilde Y_i}{\sum \tilde D_i^2}
  \]
  corresponds to a regression of residualized \(Y\) on residualized \(D\).
- The paper points out that the DML estimator can be interpreted as a linear IV estimator where the instrument is the residualized treatment \(\hat V = D-\hat m(X)\). This is noted explicitly in the discussion about ‘double prediction’ and connections to optimal instruments.:contentReference[oaicite:33]{index=33}
- The talk also highlights this “double prediction” viewpoint: one ML problem for \(Y\) given \(X\), another for \(D\) given \(X\), then regress one residual on the other.:contentReference[oaicite:34]{index=34}

---

## Slide 11 — Sample splitting and cross-fitting

- The problematic remainder terms without sample splitting involve expressions like
  \[
  \frac{1}{\sqrt{n}}\sum V_i(\hat g(X_i) - g_0(X_i)),
  \]
  where \(\hat g\) is estimated on the same sample, so the dependence structure can cause these terms to fail to vanish at the desired rate.:contentReference[oaicite:35]{index=35}
- The paper explains that with sample splitting, we estimate nuisances on an auxiliary sample, and conditionally on that sample, terms like the above have mean zero and variance determined by the squared nuisance error. The variance then goes to zero as the nuisance error shrinks.:contentReference[oaicite:36]{index=36}
- Cross-fitting (swapping main and auxiliary samples or using K-fold versions) restores efficiency while maintaining these favorable properties. This is formalized via DML1 and DML2 in Section 3, but in the slides we avoid naming those explicitly to keep things lighter.:contentReference[oaicite:37]{index=37}

---

## Slide 12–13 — Algorithm for PLR DML

- Definitions 3.1 and 3.2 give formal DML algorithms:
  - DML1 averages fold-specific solutions of orthogonal estimating equations.
  - DML2 defines one global estimator solving a single pooled orthogonal estimating equation built from cross-fitted nuisances.:contentReference[oaicite:38]{index=38}
- The slides give a simplified operational version: K-fold split, fit nuisances out-of-fold, construct residuals, regress residualized outcome on residualized treatment per fold, then average.
- Variance estimation uses the empirical influence function of the orthogonal score; Theorem 3.1 and 3.2 provide asymptotic linearity and show how to estimate the asymptotic variance.:contentReference[oaicite:39]{index=39}

---

## Slide 14 — What double ML delivers (high level)

- Theorem 3.1 and 3.3 show that under approximate Neyman orthogonality and appropriate rate conditions on the nuisance estimators, DML estimators are asymptotically linear and Gaussian:
  \[
  \sqrt{n}(\hat\theta - \theta_0) =
    \frac{1}{\sqrt{n}}\sum \varphi(W_i) + o_p(1),
  \]
  where \(\varphi\) is the influence function.:contentReference[oaicite:40]{index=40}
- Under homoscedasticity and with efficient scores, Corollary 3.2 notes that DML achieves the semiparametric efficiency bound. For the PLR model, the orthogonal score we used is efficient under homoscedasticity.:contentReference[oaicite:41]{index=41}

---

## Slide 15 — Simulation: prediction vs causal estimation

- The simulations in the paper and in the talk consider designs where \(g_0\) is particularly suitable for random forests. Figure 1 in the paper compares the conventional (non-orthogonal) ML estimator with the DML estimator:
  - The conventional estimator’s histogram is biased and poorly approximated by a normal distribution.
  - The DML estimator’s histogram is centered at zero and well approximated by its normal limit.
- The talk also presents Monte Carlo results illustrating this contrast, emphasizing that prediction performance and causal performance can diverge.:contentReference[oaicite:43]{index=43}

---

## Slide 16 — Application: 401(k) eligibility and savings

- Section 6.2 of the paper provides an application to 401(k) eligibility and net financial assets. The outcome and covariates are defined as in the slide:
  - Net financial assets include IRAs, 401(k), checking, bonds, stocks, mutual funds, minus non-mortgage debt.
  - Covariates include age, income, family size, education, marital status, two-earner status, DB pension status, IRA participation, home ownership.:contentReference[oaicite:44]{index=44}
- The identification strategy follows Poterba et al. (1994): conditional on these covariates, eligibility can be treated as exogenous.
- The paper reports DML estimates of the ATE of eligibility on assets, with magnitudes in the several thousand dollars range, robust across a variety of learners. The exact numbers depend on the specification but are in the \(\$7\)–\(\$9\)k ballpark.:contentReference[oaicite:45]{index=45}

---

## Slide 17 — Beyond PLR: ATE, ATTE, PLIV, LATE

**1. Interactive / nonseparable model**

- General form used in the paper’s “interactive” treatment model:
  \[
  Y = g_0(D,X) + U,\quad \mathbb{E}[U\mid X,D]=0,
  \]
  where \(D\in\{0,1\}\) is a binary treatment and \(g_0\) is an unknown function. :contentReference[oaicite:2]{index=2}  

- This nests the partially linear case:
  \[
  g_0(D,X) = \theta_0 D + g_0^{\text{PLR}}(X),
  \]
  but allows fully nonlinear interactions between \(D\) and \(X\) (treatment effect heterogeneity).

- Target parameters:
  - ATE: \(\theta_0 = \mathbb{E}[g_0(1,X) - g_0(0,X)]\).  
  - ATTE: \(\theta_0 = \mathbb{E}[g_0(1,X) - g_0(0,X)\mid D=1]\). :contentReference[oaicite:3]{index=3}  

- Orthogonal scores (high‑level):
  - For ATE, the score combines:
    - Outcome regression \(g_0(d,X)\),
    - Propensity score \(m_0(X) = \mathbb{E}[D\mid X]\),
    in a doubly‑robust form (Augmented IPW–type moment). :contentReference[oaicite:4]{index=4}  
  - Neyman orthogonality: the Gateaux derivative of the moment w.r.t. the nuisance triple \((g_0(0,\cdot), g_0(1,\cdot), m_0)\) is zero at the truth, so first‑order ML errors cancel.

- DML procedure:
  1. Split into folds; on each training fold, estimate \(g_0(0,\cdot)\), \(g_0(1,\cdot)\), and \(m_0\) with flexible ML (lasso, forests, boosting, nets, etc.).
  2. On the held‑out fold, plug these estimates into the orthogonal score for ATE/ATTE and solve the empirical moment for \(\theta\).
  3. Cross‑fit over folds and average. Under mild rate conditions (each nuisance estimated at \(o(n^{-1/4})\) in \(L_2\)), the resulting \(\hat\theta\) is \(\sqrt{n}\)-consistent and asymptotically normal. :contentReference[oaicite:5]{index=5}  

---

**2. Partially Linear IV (PLIV) model**

- Structural equations:
  \[
  Y = \theta_0 D + g_0(X) + U,\quad \mathbb{E}[U\mid Z,X]=0,
  \]
  \[
  D = m_0(Z,X) + V,\quad \mathbb{E}[V\mid Z,X]=0,
  \]
  where \(Z\) contains instruments (possibly with \(X\)) and \(D\) is endogenous. :contentReference[oaicite:6]{index=6}  

- Target parameter: \(\theta_0\) is a low‑dimensional structural parameter (e.g. IV slope).

- Orthogonal score idea:
  - Build a moment that looks like a residualised IV:
    - Outcome residual: \(Y - g_0(X) - \theta D\),
    - Instrument residual: something like \(Z - \mathbb{E}[Z\mid X]\) or \(m_0(Z,X) - \mathbb{E}[m_0(Z,X)\mid X]\), depending on the parametrisation.  
  - Arrange the score so that its derivative w.r.t. the nuisance vector \((g_0, m_0, \ldots)\) is zero at \((\theta_0,\eta_0)\) (Neyman orthogonality), so regularisation bias in \(g_0, m_0\) only enters at second order. :contentReference[oaicite:7]{index=7}  

- Algorithm pattern (PLIV version of DML):
  1. On training folds, fit ML for:
     - \(g_0(X) \approx \hat g(X)\),
     - \(m_0(Z,X) \approx \hat m(Z,X)\).
  2. On the held‑out fold, form residuals and compute the orthogonal IV‑style score.
  3. Solve the moment for \(\theta\); repeat across folds and average (DML2).

- Under regularity and IV relevance/overlap assumptions, DML achieves:
  - \(\sqrt{n}\)-rate for \(\hat\theta\),
  - Asymptotic normality with variance given by the efficient score in the homoscedastic case (PLIV example in the paper). :contentReference[oaicite:8]{index=8}  

---

**3. Intuition you can use in Q&A**

- The nonseparable model slide is there to say: *you don’t have to commit to linear treatment effects*. DML lets you estimate objects like ATE/ATTE in fully nonlinear \(Y = g_0(D,X)+U\) models by combining outcome regression and propensity scores inside an orthogonal moment.

- The PLIV bullet is to point out that DML also plays nicely with IV: once you can write down an orthogonal moment (usually a residualised version of 2SLS), you can freely plug in ML for the nuisance functions and still get valid inference for \(\theta_0\).

- If someone asks for more detail, you can refer back to:
  - Section 4 of the paper: partially linear IV with DML. :contentReference[oaicite:9]{index=9}  
  - Section 5: ATE/ATTE/LATE in the interactive model. :contentReference[oaicite:10]{index=10}  

---

## Slide 18 — What DML does not fix

- The paper repeatedly stresses that DML works under the same identification assumptions as the underlying causal models:
  - PLR: conditional exogeneity given \(X\).
  - ATE/ATTE: unconfoundedness given \(X\).
  - PLIV/LATE: valid instruments with appropriate structural assumptions.:contentReference[oaicite:49]{index=49}
- DML does not address unobserved confounding; it only allows flexible nonparametric estimation of the observed-nuisance structure. The need for good research design and careful control selection is highlighted especially in the applications section.:contentReference[oaicite:50]{index=50}

---

## Slide 19 — Evidence from method evaluation studies

- While the specific ‘Estimating Causal Effects with Double Machine Learning – A Method Evaluation’ paper is not in the uploaded files, its findings are consistent with broader simulation evidence:
  - DML with rigid learners (e.g.\ plain lasso in misspecified designs) can inherit bias similar to OLS.
  - DML with flexible learners tends to reduce bias in nonlinear confounding scenarios.
- The DML paper’s empirical and simulation results show that choice of learner inside DML affects performance: see the 401(k) example where different ML methods give similar but not identical estimates, and the simulation figures where non-orthogonal estimators perform poorly even with strong predictors.
- This supports the message that DML is not a magic bullet: ML choice heavily influences finite-sample performance.

---

## Slide 20 — Checklist & take-home message

- The general abstract theory in Sections 3 and 5 shows that as long as nuisance estimators achieve certain rates and orthogonality holds, DML yields valid asymptotic inference for low-dimensional \(\theta_0\).:contentReference[oaicite:52]{index=52}
- In practice, the workflow ‘identify estimand, derive orthogonal score, choose ML, cross-fit, do sensitivity checks’ is exactly what the paper suggests via its repeated pattern across models (PLR, PLIV, ATE/ATTE, LATE).
- Victor Chernozhukov’s talk also emphasizes this as a reusable template: two prediction problems for nuisances, one orthogonal estimating equation, sample splitting and averaging.:contentReference[oaicite:53]{index=53}

---

## Meta: talk-structure choices vs. guidelines

- The structure of the talk (early statement of key idea, emphasis on a single central idea, use of one running model with one main algorithm) follows typical advice for research talks:
  - Communicate one key idea; do not try to present all technical details.:contentReference[oaicite:54]{index=54}
  - Prefer depth over breadth, and use examples (PLR, 401(k)) to ground intuition.
  - Keep must-have parts (motivation, key result, main example) within the allotted time, and be ready to truncate advanced material if needed.

These notes should put you in a good position to answer deeper questions about:

- why naive plug-in fails (\(\sqrt{n}\)-bias decomposition),
- how orthogonality is defined and checked,
- why cross-fitting is necessary,
- how the method extends beyond PLR,
- and what the empirical implications are.

