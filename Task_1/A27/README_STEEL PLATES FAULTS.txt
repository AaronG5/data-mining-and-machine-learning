STEEL PLATES FAULTS duomenų aibė

Visi variantai parengti iš vartotojo pateikto originalaus UCI Steel Plates Faults failo.
Svarbu:
1. Prieš duomenų analizę patikrinti duomenų tipus ir reikšmių formatą.
2. Patikrinti praleistas reikšmes, dublikatus, logiškai galimas požymių reikšmių ribas ir išskirtis.
3. Nešalinti statistinių išskirčių automatiškai. Jas identifikavus patikrinti ar tai ne klaidos, pagrįsti atsakymus.
4. Visus valymo sprendimus realizuoti atkuriamu programiniu kodu.
# Steel Plates Faults – bendras duomenų aibės aprašas

Duomenys aprašo plieno plokščių paviršiaus defektus. Viena eilutė atitinka vieną aptiktą defektą. Originaliame rinkinyje yra 1 941 objektas, 27 įvesties kintamieji ir 7 defektų klasės.

**Originali UCI aibė:** https://archive.ics.uci.edu/dataset/198/steel+plates+faults
**DOI:** https://doi.org/10.24432/C5J88N

Šiuose mokomuosiuose variantuose `TypeOfSteel_A300` ir `TypeOfSteel_A400` neįtraukti, nes tai kategorinio plieno tipo 0/1 indikatoriai. 
Studentų failuose palikti 25 kiekybiniai požymiai ir `class`.

## Požymiai
- **`X_Minimum`** – Mažiausia defekto srities X koordinatė. (vaizdo koordinatė)
- **`X_Maximum`** – Didžiausia defekto srities X koordinatė. (vaizdo koordinatė)
- **`Y_Minimum`** – Mažiausia defekto srities Y koordinatė. (vaizdo koordinatė)
- **`Y_Maximum`** – Didžiausia defekto srities Y koordinatė. (vaizdo koordinatė)
- **`Pixels_Areas`** – Defekto srities plotas pikseliais. (pikseliai)
- **`X_Perimeter`** – Defekto kontūro charakteristika X kryptimi. (pikselių / vaizdo ilgio vienetai)
- **`Y_Perimeter`** – Defekto kontūro charakteristika Y kryptimi. (pikselių / vaizdo ilgio vienetai)
- **`Sum_of_Luminosity`** – Bendra defekto srities pikselių šviesumo suma. (intensyvumo suma)
- **`Minimum_of_Luminosity`** – Mažiausias pikselio šviesumas defekto srityje. (intensyvumo skalė)
- **`Maximum_of_Luminosity`** – Didžiausias pikselio šviesumas defekto srityje. (intensyvumo skalė)
- **`Length_of_Conveyer`** – Su gamybos / konvejerio geometrija susijęs ilgio rodiklis. (ilgio vienetai)
- **`Steel_Plate_Thickness`** – Plieno plokštės storis. (originalaus rinkinio storio vienetai)
- **`Edges_Index`** – Defekto padėties / krašto indeksas. (0–1)
- **`Empty_Index`** – Defekto užpildymo / tuštumos indeksas. (0–1)
- **`Square_Index`** – Defekto kvadratiškumo indeksas. (0–1)
- **`Outside_X_Index`** – Santykinis defekto dydžio rodiklis X kryptimi. (0–1)
- **`Edges_X_Index`** – Kraštinių indeksas X kryptimi. (0–1)
- **`Edges_Y_Index`** – Kraštinių indeksas Y kryptimi. (0–1)
- **`Outside_Global_Index`** – Bendras išorinės orientacijos / dydžio indeksas. (0–1)
- **`LogOfAreas`** – Defekto ploto logaritminis deskriptorius. (log skalė)
- **`Log_X_Index`** – X krypties dydžio logaritminis deskriptorius. (log skalė)
- **`Log_Y_Index`** – Y krypties dydžio logaritminis deskriptorius. (log skalė)
- **`Orientation_Index`** – Defekto orientacijos indeksas. (maždaug -1–1)
- **`Luminosity_Index`** – Santykinis šviesumo indeksas. (originalioje aibėje maždaug -1–0.64)
- **`SigmoidOfAreas`** – Transformuotas defekto ploto rodiklis. (0–1)
- **`class`** – defekto tipas: Pastry, Z_Scratch, K_Scratch, Stains, Dirtiness, Bumps arba Other_Faults.

Originali UCI aibė neturi praleistų reikšmių. 
Prieš analizę būtina patikrinti tipus, dublikatus, praleistas reikšmes, logines požymių reikšmių ribas ir išskirtis.