# UCI HAR Activity Classification

Skelet repozitorijuma za projekat iz mašinskog učenja:
**Klasifikacija ljudskih aktivnosti** na UCI HAR skupu podataka korišćenjem logističke regresije, SVM-a i ansambl metoda.

## Plan repozitorijuma

- `data/` — ulazni i pripremljeni podaci
- `notebooks/` — EDA i eksperimenti
- `src/` — kod za pipeline (obrada podataka, feature engineering, modeli, evaluacija)
- `configs/` — konfiguracije eksperimenata
- `reports/` — izveštaji i grafici
- `tests/` — testovi

## Brzi start

```bash
git clone <repo-url>
cd uci-har-activity-classification
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Ovo je inicijalni skelet bez implementacije modela.
