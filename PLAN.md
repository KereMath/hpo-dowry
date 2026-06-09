# Cost-Aware Optimal Stopping for HPO — Araştırma & Ürün Planı

## 0. Konumlandırma (thesis statement — kilitlenecek)

**Yanlış hedef (kaybederiz):** "BO/Hyperband'den daha iyi config bulan yeni bir arama yöntemi."
**Doğru hedef (kazanırız):** Herhangi bir sıralı HPO optimizer'ının (Random Search, BO, Hyperband/ASHA)
ÜSTÜNE takılan, **maliyet-duyarlı, ayarsız (parameter-free), EVT-temelli optimal durdurma katmanı**;
sabit-bütçe işletimini **kalite-vs-maliyet Pareto cephesinde domine eder**. İleriye-bakan (meta-öğrenilmiş)
varyant, miyopik kuralın kaybettiği düşük-regret köşesini kapatır.

**Hangi boyutlarda "HPO'ları geçiyoruz" (dürüst eksenler):**
1. **Eşit kalitede toplam maliyet / wall-clock** — BO ve Hyperband'in *önceden belirlenmiş bütçe* zayıflığı var;
   ne zaman duracağına dair prensipli bir kriterleri yok. Biz onu sağlıyoruz.
2. **Stopper'ın kendisinin ayarsız olması** — patience/fixed ayar ister; biz maliyet λ'dan türetiriz.
3. **Maliyet heterojenliği farkındalığı** — config başına eğitim süresi farklı; çoğu durdurma sezgisi bunu yok sayar,
   biz wall-clock'u doğrudan optimize ederiz.

> Tek cümle: *"A universal, cost-aware, parameter-free stopping layer that Pareto-dominates fixed-budget
> operation of RS/BO/Hyperband on the quality-vs-cost frontier, with a learned forward-looking variant."*

---

## 1. Faz 1 — Altyapı & gerçek benchmark'lar (kredibilite temeli)

Şu anki 3 sklearn manzarası ispat-of-concept; tez için **standart tabular/surrogate HPO benchmark'ları** şart
(eğitim yok, binlerce config, onlarca task, maliyet/runtime kayıtlı, tekrarlanabilir):

- **LCBench** (learning curves + final), **YAHPO Gym** (surrogate, çok task + runtime), **HPO-B**,
  **HPOBench**, **NAS-Bench-201**.
- Çıktı: her benchmark için `(task_id -> {scores[], costs[], best})` standardize edilmiş bir veri katmanı.
- **Offline fallback:** internet yoksa, kendi sklearn manzaralarımızı **çok dataset × çok model ailesi**
  (RF, GBM/HistGB, SVM, MLP, LogReg) ile genişletip cache'leyelim (≥20 task) — yine de anlamlılık için yeterli.

**Done tanımı:** ≥20 task, her birinde ≥500 config (skor+maliyet), reproducible loader + cache.

---

## 2. Faz 2 — Gerçek rakipler (jürinin soracağı kıyas)

Optimizer'lar (her biri bizim stopper'ımızla sarılabilir olacak):
- Random Search, **TPE/BO** (Optuna veya SMAC3), **Hyperband/ASHA** (Ray Tune / HpBandSter).

Durdurma baseline'ları (stopper ekseninde rakipler):
- fixed-budget, patience/plateau, **1/e secretary** (no-recall ordinal), vanilla EI-threshold,
  Bayesian/curve-extrapolation early-stop (Domhan-tarzı), ve **bizim oracle** (hindsight üst-sınır).

**Done tanımı:** "optimizer-alone (bütçe taranmış)" vs "optimizer + stopper" tam matris, tek arayüzde.

---

## 3. Faz 3 — Yöntem geliştirme (asıl novelty)

- **M1 — Robust tail estimation.** Gauss-tail çöküşünü çöz (digits'te %95 bütçe felaketi). EVT/GPD'yi
  threshold-seçimi + regularizasyon + guard ile sağlamlaştır; model-seçim (EVT vs lognormal vs gauss)
  veya **guarded ensemble**; tail-belirsizliğinde *muhafazakâr* tarafa yanlan.
- **M2 — Maliyet heterojenliği.** Config-başı maliyet tahmincisi + non-uniform Weitzman (kutu-başı rezervasyon
  değeri); wall-clock optimize et, #eval değil.
- **M3 — İleriye-bakan / meta-öğrenilmiş durdurma (tavan-yükseltici, "hindsight+USP").**
  Meta-veri: çok task × oracle-stop noktaları. Online özelliklerden (gözlem-fazı dağılım istatistikleri,
  iyileşme eğimi, incumbent, tahmini maliyet, geçen-süre) **oracle durma noktasını taklit eden politika**
  öğret (gradient-boosted classifier ya da küçük RL/optimal-stopping-as-sequential-decision).
  Miyopik kuralın kaybettiği köşeyi kapatması beklenen kısım burası.
- **M4 — Teori.** Maliyet-duyarlı durdurmanın oracle'a göre **competitive ratio / regret** garantisi;
  tail-tahmin hatasından durma-suboptimalliğine PAC-tarzı sınır.

**Done tanımı:** M1 zorunlu (güvenlik); M3 ana katkı; M2/M4 güçlendirici.

---

## 4. Faz 4 — Sıkı değerlendirme (istatistiksel sağlamlık)

- **Metrikler:** cost-to-target, fixed-budget'ta final regret, **anytime regret-vs-cost AUC**,
  task-bazında normalize, **çok seed**.
- **İstatistik:** task'lar arası **Wilcoxon signed-rank**, **critical-difference (Nemenyi) diyagramı**,
  bootstrap güven aralıkları.
- **Ablasyonlar:** tail modeli, k_obs, λ duyarlılığı, meta-learning var/yok, optimizer çeşidi.
- **Belirleyici grafik:** her optimizer için, stopper'lı vs stopper'sız Pareto cephesi (lower-left daha iyi).

**Done tanımı:** "X optimizer + bizim stopper, eşit kalitede %Y±Z daha az maliyet (p<0.05, N task)" cümlesi
benchmark'lar genelinde kurulabiliyor.

---

## 5. Faz 5 — Ürünleştirme (opsiyonel paralel track)

- **Drop-in eklenti:** Optuna `StudyStopper` / Ray Tune `Stopper` / sklearn-uyumlu callback.
- "Cloud eğitim maliyetini ayarsız %X kısan durdurma katmanı" → OSS repo + küçük benchmark raporu.
- Ürün kazanım ekseni: **wall-clock / $ tasarrufu**, sıfır-ayar entegrasyon.

---

## 6. Risk & kill-criteria (her fazda dürüst kapı)

- Faz 2 sonrası: gerçek BO/Hyperband'e karşı **eşit kalitede en az başa-baş + ayar-avantajı** yoksa →
  konumlandırmayı "niş, ayarsız termination" diye daralt (yine tez olur, daha zayıf).
- M3 (meta-learning) miyopik köşeyi **kapatmazsa** → ana-konferans hedefini bırak, workshop+tez ile yetin.
- Tail M1 sağlamlaşmazsa → metodu "EVT-only, guarded" diye sınırla, gauss'u tamamen çıkar.

---

## 7. Kilometre taşları (tez ölçeği, ~aylar)

1. **MS1 (altyapı):** benchmark loader + ≥20 task + baseline optimizer'lar. → "deney koşulabilir".
2. **MS2 (sinyal):** stopper'lı vs stopper'sız Pareto, gerçek rakiplere karşı, istatistikli. → "tez iskeleti".
3. **MS3 (novelty):** M1 robust tail + M3 meta-learned stopping, köşe kapanıyor mu? → "paper'lık katkı".
4. **MS4 (yazım):** teori (M4) + ablasyonlar + figürler → tez/paper draft + (ops.) OSS sürüm.

---

## 8. Çıktılar (deliverables)

- Tez bölümleri (problem, yöntem, teori, deney, limitasyon).
- Paper draft (AutoML-Conf / workshop → koşullu ana-konferans).
- Reproducible kod + cache'lenmiş benchmark + figür üreten scriptler.
- (Ops.) Optuna/Ray-Tune eklentisi olarak OSS.

---

## 9. Hemen sıradaki adım (Faz 1 başlangıcı)

1. Benchmark veri erişimini netleştir (internet var mı? LCBench/YAHPO indirilebilir mi?).
2. Yoksa: sklearn manzara üreticisini **çok-dataset × çok-model** olacak şekilde genişlet, ≥20 task cache.
3. Baseline optimizer arayüzünü (RS/BO/Hyperband) tek API altında topla; stopper'ı "study callback" yap.
