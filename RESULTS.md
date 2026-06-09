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

## Açık işler
- Literatür taraması (novelty + related work; Makarova 2022 vd.) — koşuyor.
- M3: meta-öğrenilmiş ileriye-bakan durdurma (hindsight+USP), düşük-maliyet miyopik açığı için.
- (ops.) Hyperband/ASHA substrate; ürün: Optuna/Ray-Tune stopper eklentisi.
