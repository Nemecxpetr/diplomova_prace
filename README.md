# Synchronizace symbolického zápisu s hudebními interpretacemi odpovídající skladby

Diplomová práce na téma synchronizace symbolického zápisu s hudebními interpretacemi odpovídající skladby.

## Cíle
Prozkoumat možnosti hudební synchronizace symbolického zápisu hudby s reálnými interpretacemi daného díla.

- Zaměření se na metody 'DTW' s využitím dalších parametrů hudebních nahrávek. 
- Sestavení vlastního datasetu a otestování způsobu odhadu přesnosti hudební synchronizace.
- Vytvoření skriptů v jazyce Python, které převedou symbolický zápis, například ve formě strojového notového zápisu nebo .mid souboru, na klasický audio formát odpovídající v čase reálné interpretaci skladby.

Na závěr vyhodnocení limitace použitých metod a přesnost výsledné synchronizace. 

Cílem **semestrálního projektu** bylo:
- popis principů hudební synchronizace,
-  sestavení datasetu
-  základní implementace score-to-audio synchronizace.
 
Cílem **diplomové práce** je:
- implementace skriptů pro každou část synchronizačního procesu
- včetně výsledného testování přesnosti použitých metod
- a výsledek bude demonstrován na praktických ukázkách.

Prozatím hotovo:
- implementace skriptů pro každou část synchronizačního procesu, ale kód je messy
  - implementace knihovny MIDI handler, která obsahuje funkce pro konverzi midi na csv, panda data frame a zpět, rovněž jako funkce umožnující tuto konverzi (přepočet tempových dat, apod. )
  - implementace knihovny DTW, která obsahuje funkce pro výpočet synchronizačních dat
    - v momentální fázi obsahuje části, které zkouší různé metody, především experimentujeme s různými způsoby tvorby chroma vektorů a parametrů při jejich tvorbě (velikost vzorkovacího okna, velikost skoku okna, formát chroma vektoru, apod. ...).
    - Pro zvýšení účinnosti skriptu je také experimentálně přidána funkce ghost note, která se snaží eliminovat nepřesnosti způsobené tichem/šumem na začátku a konci skladby v audio interpretaci
       - tato funkce je na několika místech celého projektu zdvojená a je potřeba ji sjednotit (při posledním pokusu o sjednocení se skript rozbil a momentálně je nefunkční).
  - implementace skriptu DP (prozatimní název "diplomová práce"), který celý proces realizuje v jedné funkci a testuje na základě adekvátních dat z datasetu.
- částečně kompletní dataset
  - potřeba vyčisti a rozšířit
- česká verze teoretické části textu práce

Potřeba dodělat:
- vyčistit kód (sjednotit zdvojené funkce, vybrat jednotný přístup, nebo více parametrizovat různé varianty, odstarnit přebytečný kód, dopsat dokumentaci, ...)
- otestovat na širším vzroku dat
- vyhodnotit použité přístupy a jejich účinnost, atd.
- dopsat text teoretické části a napsat text praktické části


