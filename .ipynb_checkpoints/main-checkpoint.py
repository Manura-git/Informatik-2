import streamlit as st
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

# --- KONFIGURATION ---
st.set_page_config(page_title="Pandera Präsentation", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Gehe zu:", [
    "1. Einleitung & Problem",
    "2. Was ist Pandera?",
    "3. Live-Demo: Daten-Check",
    "4. Fazit & Takeaways"
])

# --- SEITE 1: EINLEITUNG ---
if selection == "1. Einleitung & Problem":
    st.title("🛡️ Datenvalidierung mit Pandera")
    st.header("Das Problem: 'Garbage In, Garbage Out'")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Warum brauchen wir das?
        * Manuelle Datenpflege ist fehleranfällig.
        * APIs oder CSVs ändern sich ohne Vorwarnung.
        * **Folge:** Abstürze oder (schlimmer) falsche Analyseergebnisse.
        """)
    with col2:
        # Hier könntest du ein Bild oder ein kaputtes DataFrame anzeigen
        st.warning("Beispiel für kaputte Daten: Preis = 'kostenlos' (statt 0.0)")

# --- SEITE 2: WAS IST PANDERA? ---
elif selection == "2. Was ist Pandera?":
    st.title("Was ist Pandera?")
    st.markdown("""
    Pandera ist eine statistische Typisierung für Pandas DataFrames.

    * **Schema-Definition:** Wir legen fest, wie Daten auszusehen haben.
    * **Informativ:** Präzise Fehlermeldungen statt `KeyError`.
    * **Flexibel:** Einfache Integration in bestehende Pipelines.
    """)

    st.code("""
# So sieht ein Schema aus:
schema = DataFrameSchema({
    "Alter": Column(int, Check.in_range(0, 120)),
    "Status": Column(str, Check.isin(["aktiv", "inaktiv"]))
})
    """, language="python")

# --- SEITE 3: LIVE-DEMO ---
elif selection == "3. Live-Demo: Daten-Check":
    st.title("🔍 Interaktive Live-Demo")

    # Hier kommt der interaktive Teil aus dem vorherigen Vorschlag rein
    df = pd.DataFrame({"Alter": [25, 150], "Status": ["aktiv", "unbekannt"]})
    edited_df = st.data_editor(df)

    if st.button("Validieren"):
        # Validierungslogik...
        st.write("Ergebnis der Prüfung...")

# --- SEITE 4: FAZIT ---
elif selection == "4. Fazit & Takeaways":
    st.title("Zusammenfassung")
    st.success("""
    1. **Sicherheit:** Schemas schützen vor Datenfehlern.
    2. **Dokumentation:** Das Schema ist gleichzeitig die Doku.
    3. **Effizienz:** Weniger Zeit für Debugging, mehr für Analyse.
    """)
    st.balloons()
