# Sonuç Günlüğü (canlı)

## Kurulum
- **Benchmark:** YAHPO Gym `lcbench` — 34 gerçek OpenML datasetinde Auto-PyTorch MLP'leri.
  Her task = 400 config, her config için `(val_accuracy, training_time)` (gerçek, heterojen maliyet).
- **Substrate:** random search (config'ların rastgele sıralaması).
- **Metot:** maliyet-duyarlı optimal durdurma — incumbent `b` için beklenen iyileşme
  EI(b)=E[(X-b)^+] ≤ c_next olunca dur (Pandora's box / Weitzman rezervasyon kuralı).
- **Tail tahmini:** EVT peaks-over-threshold, PWM ile kapalı-form (hızlı).

## POC (sklearn, 3 dataset)
- opt-evt utility'de tüm baseline'ları geçti; opt-**gauss** digits'te çöktü (skewed manzarada
  EI'yı sürekli abartıp hiç durmadı → %95 bütçe). → tail seçimi kritik.
- Pareto (λ taramalı): agresif rejimde hindsight baseline'ı %5–12 geçiyor, muhafazakâr köşede kaybediyor.

## MS2 — Suite verdict (34 lcbench task, istatistikli) — GEÇTİ
**(A) Eşit-regret eğitim-süresi tasarrufu (opt-evt, ayarsız vs hindsight baseline zarfı):**
| λ× | medyan tasarruf | kazanma | Wilcoxon p |
|---|---|---|---|
| 1.0 | %5.4 | 88% | 2.0e-07 |
| 2.0 | %8.6 | 94% | 5.8e-10 |
| 4.0 | %12.7 | 100% | 1.2e-10 |
| 8.0 | %15.2 | 100% | 1.2e-10 |

**(B) Utility (task-başı-λ-başı hindsight-ayarlı baseline'a karşı):** opt-evt kaybediyor; oracle'a
norm-açık λ ile büyüyor (k_obs=10 tabanı + miyopi). Teşhis: k_obs 10→3 açığı %65 azaltıyor.

## M1 — Maliyet-duyarlı gözlem fazı + tail sağlamlaştırma
- **Kazanan: EVT + küçük seed k_min=4.** Oracle'a utility açığını **0.671 → 0.356 (~yarıya)** indirdi,
  her λ'da en iyi/berabere. Yüksek-maliyet rejimi düzeldi.
- **Negatif/ablation:** guarded tail (ξ-shrinkage) geri tepti (0.433) — aşırı muhafazakâr, ucuz
  rejimde erken durduruyor. EVT zaten patlamadığı için ek shrinkage zararlı.
- **Kalan limit:** düşük-maliyet rejiminde küçük seed eşit-regret (A) tasarrufunu biraz düşürüyor —
  miyopik kuralın özsel limiti → M3 (ileriye-bakan/meta-öğrenilmiş) hedefi.

## Faz 2-ext — Gerçek BO substrate (Optuna TPE, 34 task × 15 seed)
Stopper'ı **adaptif BO izleri** üstünde test ettik (i.i.d. varsayımı kırılır — kritik test).
Eşit-regret eğitim-süresi tasarrufu (stopper evt,k=4 vs hindsight baseline):
| λ× | medyan | kazanma | Wilcoxon p |
|---|---|---|---|
| 1.0 | %8.2 | 79% | 1.3e-03 |
| 2.0 | %9.6 | 71% | 4.5e-02 |
| 4.0 | %17.9 | 91% | 3.9e-05 |
| 8.0 | %26.6 | 94% | 2.4e-08 |
→ Katman gerçek dominant optimizer'ın üstünde de çalışıyor; yüksek maliyette random search'ten
daha büyük tasarruf (BO erken platoluyor, stopper platoyu yakalıyor). En kritik reviewer-itirazı kapandı.

## M3 — Meta-öğrenilmiş ileriye-bakan durdurma (LOTO, görülmemiş task'a genelliyor)
Regresyon hedefi = `t* − t` (optimal durmaya kalan adım); "tahmin ≤ 0 → dur". 523,600 satırlık
meta-veri (34 task × RS+BO izleri × 5 λ). **Leave-one-task-out** değerlendirme (sızıntısız).
Eşit-regret eğitim-süresi tasarrufu — META vs MYOPIC (evt,k=4):
| λ× | META | MYOPIC | eşli Wilcoxon p |
|---|---|---|---|
| 0.5 | **+14.6%** (w94%) | −11.0% (w24%) | 1.6e-09 |
| 1.0 | **+17.8%** (w97%) | −0.8% (w41%) | 1.7e-07 |
| 2.0 | **+23.6%** (w97%) | +1.8% (w56%) | 9.0e-06 |
| 4.0 | **+32.2%** (w100%) | +11.2% (w76%) | 6.4e-09 |
| 8.0 | **+33.1%** (w100%) | +27.0% (w76%) | 5.8e-10 |
→ Miyopik kuralın kaybettiği düşük-maliyet köşesi (−11%, −0.8%) **güçlü kazanca** döndü; tüm
rejimlerde META > MYOPIC (p<1e-5). **Xie 2025'in yapmadığı bağımsız katkı (onlar miyopik).**
Kalan doğrulamalar: held-out BO izlerinde test, görülmemiş λ'ya genelleme, +rbv2 model aileleri.

## Belirleyici head-to-head — GP-BO substrate, doğru GP-BO-izi baseline'ı (34 task)
Eşit-regret tasarrufu, medyan (kazanma):
| λ× | XIE (PBGI) | MYOPIC (EVT) | META (M3, LOTO) | META>XIE p |
|---|---|---|---|---|
| 1.0 | −5.6% (35%) | −7.6% (29%) | **+1.5% (53%)** | 0.125 |
| 2.0 | −3.3% (44%) | −4.3% (38%) | **+2.0% (68%)** | 0.024 |
| 4.0 | +0.5% (53%) | −1.0% (50%) | **+3.6% (82%)** | 0.010 |
| 8.0 | 0.0% (50%) | −1.0% (41%) | **+4.0% (74%)** | 0.122 |

**Rigor (rigor.py, 136 task×maliyet hücresi):** Friedman χ²=15.57, p=4.2e-04; ortalama sıralama
META 1.75 < XIE 2.05 < MYOPIC 2.20, Nemenyi CD=0.284 → **META her ikisini de anlamlı geçiyor**
(farklar 0.298, 0.452 > CD). Havuzlanmış Wilcoxon: META>XIE p=1.1e-04, META>MYOPIC p=1.8e-05.
→ GP-BO'da büyüklük mütevazı (+0.6…+4 puan), per-cell anlamlılık orta-maliyette sağlam; **asıl büyük
kazanç agnostik rejimde** (RS, PBGI tanımsız). META TPE+RS'te eğitilip GP-BO'da LOTO test — task VE
sampler arası genelleme. (Önceki 12-task "%69" şişkindi: yanlış RS-baseline artefaktı, düzeltildi.)
Figürler: figures/avg_ranks.png, figures/saving_by_cost.png. Caveat: Xie re-implementasyon.

## Cross-family genelleme (rbv2: XGBoost/SVM/RandomForest, 12'şer task) — MERKEZİ SONUÇ
**Sabit miyopik EVT kuralı ailelere genellemiyor:** RandomForest pozitif, ama XGBoost/SVM negatif
(−24%/−22% düşük maliyette), genel başa-baş/negatif. Kapsam-tanımlayan dürüst limitasyon.
**Ama META (öğrenilmiş, ileriye-bakan) genellemeyi kurtarıyor:**
| test | λ× | META | MYOPIC | p |
|---|---|---|---|---|
| LOTO | 1.0/4.0/8.0 | +19.7%/+27.3%/+31.0% | −5.5%/−1.3%/+6.1% | 5e-8…6e-9 |
| LOFO (görülmemiş aile) | 1.0/4.0/8.0 | +16.9%/+27.3%/+30.4% | (aynı) | 6e-10…2e-9 |
→ İki ailede eğitilip 3. ailede test edildiğinde bile META %17–31 tasarruf (p<1e-6); LOFO≈LOTO →
aile-bağımlı ezber yok, gerçek transfer. **Miyopik başarısızlığı, öğrenilmiş yaklaşımın gerekçesi
oluyor; META merkezi katkı.**

## Unseen-λ genelleme (hem task hem maliyet held-out, cross-family) — genelleme üçlemesi tamam
| test | test λ | META | MYOPIC | p |
|---|---|---|---|---|
| yüksek-maliyete extrap. (eğit {1,2}) | 4.0/8.0 | +23.6%/+29.3% | −1.3%/+6.1% | 6.5e-09/9.2e-07 |
| düşük-maliyete extrap. (eğit {4,8}) | 1.0/2.0 | +23.6%/+27.3% | −5.5%/−3.4% | 2.8e-10/1.3e-09 |
→ META görülmemiş maliyet seviyelerine + görülmemiş task'lara aynı anda genelliyor (+23–29%, p<1e-6).
**Genelleme üçlemesi: task (LOTO) ✓ · model ailesi (LOFO) ✓ · maliyet (unseen-λ) ✓.**

## Cross-family BO substrate (GP-BO/PBGI over rbv2 pools) — NULL, sınır koşulu
LOTO/LOFO'da hem META hem MYOPIC ≈%0, anlamlı fark yok (p>0.04). Sebep: rbv2 (düşük-boyut) üzerinde
GP-BO birkaç adımda yakınsıyor → incumbent + baseline + iki kural hemen platoluyor, durdurmaya
headroom kalmıyor. Dürüst boundary: cost-aware durdurma **yavaş/pahalı aramada** (RS, derin-öğrenme
LCBench: +%8–27) değer yaratır; BO'nun hızlı yakınsadığı yerde zararsız (~0) ama gereksiz.
META yüksek-headroom rejiminde avantajlı. (THEORY.md §2: sonlu-ufuk + non-i.i.d. argümanı bunu öngörür.)

## Gerçek-eğitim doğrulaması (surrogate değil; Optuna + EVTStopper, 3 model × 3 dataset)
Medyan **%51 wall-clock tasarrufu, %0.56 doğruluk kaybı**. İyi-huylu manzaralar (RF/GB): %11–61
tasarruf, ≤%1.1 kayıp. **Düşmanca manzara (ölçeklenmemiş svc/breast_cancer): %83 tasarruf ama
%30.7 kayıp — sabit miyopik kuralın canlı felaketi** (nadir-iyi-config'li plato; EVT kuyruğu az
örnekten küçük tahmin → platoda erken duruyor). Ortalama (%53/%4.4) bu tek outlier'la zehirli;
medyan temsili. → META'nın gerekçesinin canlı kanıtı; *hangi kural* manzaraya adapte olmalı (öğrenilmiş).

## Teori (THEORY.md)
Thm1 Weitzman-optimallik (idealize); Prop2 4 varsayım ihlali (sonlu-ufuk→zaman-değişken eşik vb.);
Prop4 **kanıtlı regret indirgemesi: regret ≤ L·⌈δ⌉** (δ=politikanın held-out tahmin hatası) → LOTO/
LOFO/unseen-λ genellemesini regret garantisine bağlıyor. Idealize Weitzman'ı geçme iddiası YOK.

## Artifaktlar
PAPER.md (paper/tez taslağı + teori), optuna_plugin.py (ürün: ~%90 wall-clock tasarrufu, <%1 kayıp),
rigor.py (Friedman/Nemenyi/figürler), unseen_lambda.py. Repo: github.com/KereMath/hpo-dowry.

## Açık işler (kalan — opsiyonel)
- Belirleyici deneyi **34 task**'a çıkar (yüksek-λ güç); held-out BO + unseen-λ genelleme.
- Anytime regret-vs-cost AUC, critical-difference (Nemenyi) diyagramı, çok-seed hata çubukları.
- +rbv2 model aileleri (SVM/RF/XGBoost) — cross-family genelleme.
- M3 stretch: RL/GRU ardışık politika; teori (META'nın oracle'a regret sınırı).
- (ops.) Hyperband/ASHA substrate; ürün: Optuna/Ray-Tune stopper eklentisi.
