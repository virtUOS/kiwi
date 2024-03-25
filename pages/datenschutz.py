import streamlit as st

st.set_page_config(page_title="Datenschutzerklärung", layout="wide", page_icon="🥝")

# st.page_link("start.py")

st.title("Datenschutzerklärung")

st.markdown("""---""")

st.header("Datenschutz auf einen Blick")
st.subheader("Allgemeine Hinweise")

st.markdown("Die folgenden Hinweise geben einen einfachen Überblick darüber, was mit Ihren personenbezogenen Daten "
            "passiert, wenn Sie unsere Website besuchen. Personenbezogene Daten sind alle Daten, mit denen Sie "
            "persönlich identifiziert werden können. Ausführliche Informationen zum Thema Datenschutz entnehmen Sie "
            "unserer unter diesem Text aufgeführten Datenschutzerklärung.")

st.subheader("Datenerfassung auf unserer Website")

st.markdown("**Wer ist verantwortlich für die Datenerfassung auf dieser Website?**")
st.markdown("Die Datenverarbeitung auf dieser Website erfolgt durch den Websitebetreiber. Dessen Kontaktdaten können "
            "Sie dem Impressum dieser Website entnehmen.")
st.markdown("**Wie erfassen wir Ihre Daten?**")
st.markdown("Ihre Daten werden zum einen dadurch erhoben, dass Sie uns diese mitteilen. Hierbei kann es sich zum "
            "Beispiel um Daten handeln, die Sie in ein Kontaktformular eingeben.")
st.markdown("Andere Daten werden automatisch beim Besuch der Website durch unsere IT-Systeme erfasst. Das sind vor "
            "allem technische Daten (zum Beispiel Internetbrowser, Betriebssystem oder Uhrzeit des Seitenaufrufs). "
            "Die Erfassung dieser Daten erfolgt automatisch, sobald Sie unsere Website betreten.")
st.markdown("**Wofür nutzen wir Ihre Daten?**")
st.markdown("Ein Teil der Daten wird erhoben, um eine fehlerfreie Bereitstellung der Website zu gewährleisten. Andere "
            "Daten können zur Analyse Ihres Nutzerverhaltens verwendet werden.")
st.markdown("**Welche Rechte haben Sie bezüglich Ihrer Daten?**")
st.markdown("Sie haben jederzeit das Recht unentgeltlich Auskunft über Herkunft, Empfänger und Zweck Ihrer "
            "gespeicherten personenbezogenen Daten zu erhalten. Sie haben außerdem ein Recht, die Berichtigung, "
            "Sperrung oder Löschung dieser Daten zu verlangen. Hierzu sowie zu weiteren Fragen zum Thema Datenschutz "
            "können Sie sich jederzeit unter der im Impressum angegebenen Adresse an uns wenden. Des Weiteren "
            "steht Ihnen ein Beschwerderecht bei der zuständigen Aufsichtsbehörde zu.")
st.markdown('Außerdem haben Sie das Recht, unter bestimmten Umständen die Einschränkung der Verarbeitung Ihrer '
            'personenbezogenen Daten zu verlangen. Details hierzu entnehmen Sie der Datenschutzerklärung unter '
            '„Recht auf Einschränkung der Verarbeitung“.')

st.markdown("""---""")

st.header("Hinweise und Pflichtinformationen")

st.subheader("Datenschutz")

st.markdown("Die Betreiber dieser Seiten nehmen den Schutz Ihrer persönlichen Daten sehr ernst. Wir behandeln Ihre "
            "personenbezogenen Daten vertraulich und entsprechend der gesetzlichen Datenschutzvorschriften sowie "
            "dieser Datenschutzerklärung.")
st.markdown("Wenn Sie diese Website benutzen, werden **keine** personenbezogene Daten erhoben. Personenbezogene Daten "
            "sind Daten, mit denen Sie persönlich identifiziert werden können. Die vorliegende Datenschutzerklärung "
            "erläutert, welche Daten wir erheben und wofür wir sie nutzen. Sie erläutert auch, wie und zu welchem "
            "Zweck das geschieht.")
st.markdown("Für die Nutzung des UOS KI-Portal werden folgende Daten abgefragt und gespeichert:")
st.markdown("-  Statusgruppe und Heimateinrichtung im LDAP\n"
            "-  In einer Session verbrauchte Tokens (Abrechnungseinheiten) von OpenAI")
st.markdown("Diese Daten werden systematisch ausgewertet und genutzt, um Prognosen über die künftige Nutzung zu "
            "erhalten und Budgets den Lehreinheiten zuweisen zu können. ")
st.markdown("Wir weisen darauf hin, dass die Datenübertragung im Internet (zum Beispiel) bei der Kommunikation per "
            "E-Mail) Sicherheitslücken aufweisen kann. Ein lückenloser Schutz der Daten vor dem Zugriff durch Dritte "
            "ist nicht möglich.")
st.markdown("Das UOS KI-Portal nutzt eingebettete Dienste anderer Anbieter. Diese werden ggf. außerhalb der "
            "EU gehostet. Dies ist aktuell die API von OpenAI. ")
st.markdown("Das UOS KI-Portal übermittelt keine personenbezogenen Daten, wie ihren Usernamen oder ihre IP-Adresse an "
            "OpenAI. Die Anfragen, die Sie in der Anwendung stellen werden aber an OpenAI übertragen und dort auch "
            "gespeichert. Bitte stellen Sie sicher, dass sie keine vertraulichen Daten oder Daten, die die "
            "Persönlichkeitsrechte Dritter verletzen über das UOS KI-Portal an Open AI übermitteln. ")
st.markdown("Es gelten für die an OpenAI übermittelten Eingaben die Privacy Policy "
            "[https://openai.com/policies/privacy-policy](https://openai.com/policies/privacy-policy) von OpenAI und "
            "die durch die Universität geschlossene Datenverarbeitungs-Zusatzvereinbarung "
            "[https://openai.com/policies/data-processing-addendum]"
            "(https://openai.com/policies/data-processing-addendum)")
st.markdown("Ihre Eingaben werden nicht auf Servern der Universität Osnabrück gespeichert.")

st.markdown("""---""")

st.header("Hinweis zur verantwortlichen Stelle")

st.markdown("Universität Osnabrück\n\n"
            "Zentrum für Digitale Lehre, Campus Management und Hochschuldidaktik (virtUOS)\n\n"
            "Heger-Tor-Wall 12\n\n"
            "49074 Osnabrück\n\n"
            "Telefon: +49 541 969-6666\n\n"
            "virtuos@uni-osnabrueck.de")
st.markdown("Verantwortliche Stelle ist die natürliche oder juristische Person, die allein oder gemeinsam mit anderen "
            "über die Zwecke und Mittel der Verarbeitung von personenbezogenen Daten (z.B. Namen, E-Mail-Adressen "
            "oder ähnliches) entscheidet.")

st.markdown("""---""")

st.header("Widerruf Ihrer Einwilligung zur Datenverarbeitung")

st.markdown("Viele Datenverarbeitungsvorgänge sind nur mit Ihrer ausdrücklichen Einwilligung möglich. Sie können eine "
            "bereits erteilte Einwilligung jederzeit widerrufen. Dazu reicht eine formlose Mitteilung per Email an uns."
            " Die Rechtmäßigkeit der bis zum Widerruf erfolgten Datenverarbeitung bleibt vom Widerruf unberührt.")
st.markdown("Widerspruchsrecht gegen die Datenerhebung in besonderen Fällen sowie gegen Direktwerbung (Art. 21 DSGVO)")
st.markdown("**Wenn die Datenverarbeitung auf Grundlage von Art. 6 Abs. 1 lit. e oder f DSGVO erfolgt, haben Sie "
            "jederzeit das Recht, aus Gründen, die sich aus Ihrer besonderen Situation ergeben, gegen die Verarbeitung "
            "Ihrer personenbezogenen Daten Widerspruch einzulegen; dies gilt auch für ein auf diese Bestimmungen "
            "gestütztes Profiling. Die jeweilige Rechtsgrundlage, auf denen eine Verarbeitung beruht, entnehmen Sie "
            "dieser Datenschutzerklärung. Wenn Sie Widerspruch einlegen, werden wir Ihre betroffenen personenbezogenen "
            "Daten nicht mehr verarbeiten, es sei denn, wir können zwingende schutzwürdige Gründe für die Verarbeitung "
            "nachweisen, die Ihre Interessen, Rechte und Freiheiten überwiegen oder die Verarbeitung dient der "
            "Geltendmachung, Ausübung oder Verteidigung von Rechtsansprüchen (Widerspruch nach Art. 21 Abs. 1 "
            "DSGVO).**")
st.markdown("**Werden Ihre personenbezogenen Daten verarbeitet, um Direktwerbung zu betreiben, so haben Sie das Recht, "
            "jederzeit Widerspruch gegen die Verarbeitung Sie betreffender personenbezogener Daten zum Zwecke "
            "derartiger Werbung einzulegen; dies gilt auch für das Profiling, soweit es mit solcher Direktwerbung in "
            "Verbindung steht. Wenn Sie widersprechen, werden Ihre personenbezogenen Daten anschließend nicht mehr zum "
            "Zwecke der Direktwerbung verwendet (Widerspruch nach Art. 21 Abs. 2 DSGVO).**")

st.markdown("""---""")

st.header("Beschwerderecht bei der zuständigen Aufsichtsbehörde")

st.markdown("Im Falle von Verstößen gegen die DSGVO steht den Betroffenen ein Beschwerderecht bei einer "
            "Aufsichtsbehörde, insbesondere in dem Mitgliedstaat ihres gewöhnlichen Aufenthalts, ihres Arbeitsplatzes "
            "oder des Orts des mutmaßlichen Verstoßes zu. Das Beschwerderecht besteht unbeschadet anderweitiger "
            "verwaltungsrechtlicher oder gerichtlicher Rechtsbehelfe.")

st.markdown("""---""")

st.header("Recht auf Datenübertragbarkeit")

st.markdown("Sie haben das Recht, Daten, die wir auf Grundlage Ihrer Einwilligung oder in Erfüllung eines Vertrags "
            "automatisiert verarbeiten, an sich oder an einen Dritten in einem gängigen, maschinenlesbaren Format "
            "aushändigen zu lassen. Sofern Sie die direkte Übertragung der Daten an einen anderen Verantwortlichen "
            "verlangen, erfolgt dies nur, soweit es technisch machbar ist.")

st.markdown("""---""")

st.header("SSL- bzw. TLS-Verschlüsselung")

st.markdown('Diese Seite nutzt aus Sicherheitsgründen und zum Schutz der Übertragung vertraulicher Inhalte, wie zum '
            'Beispiel Suchanfragen, die Sie an uns als Seitenbetreiber senden, eine SSL-bzw. TLS-Verschlüsselung. '
            'Eine verschlüsselte Verbindung erkennen Sie daran, dass die Adresszeile des Browsers von “http://” auf '
            '“https://” wechselt und an dem Schloss-Symbol in Ihrer Browserzeile.')

st.markdown("""---""")

st.header("Auskunft, Sperrung, Löschung und Berichtigung")

st.markdown("Sie haben im Rahmen der geltenden gesetzlichen Bestimmungen jederzeit das Recht auf unentgeltliche "
            "Auskunft über Ihre gespeicherten personenbezogenen Daten, deren Herkunft und Empfänger und den Zweck der "
            "Datenverarbeitung und ggf. ein Recht auf Berichtigung, Sperrung oder Löschung dieser Daten. Hierzu sowie "
            "zu weiteren Fragen zum Thema personenbezogene Daten können Sie sich jederzeit unter der im Impressum "
            "angegebenen Adresse an uns wenden.")

st.markdown("""---""")

st.header("Recht auf Einschränkung der Verarbeitung")

st.markdown("Sie haben das Recht, die Einschränkung der Verarbeitung Ihrer personenbezogenen Daten zu verlangen. "
            "Hierzu können Sie sich jederzeit unter der im Impressum angegebenen Adresse an uns wenden. Das Recht "
            "auf Einschränkung der Verarbeitung besteht in folgenden Fällen:")

st.markdown("-  Wenn Sie die Richtigkeit Ihrer bei uns gespeicherten personenbezogenen Daten bestreiten, benötigen wir "
            "in der Regel Zeit, um dies zu überprüfen. Für die Dauer der Prüfung haben Sie das Recht, die "
            "Einschränkung der Verarbeitung Ihrer personenbezogenen Daten zu verlangen.\n"
            "-  Wenn die Verarbeitung Ihrer personenbezogenen Daten unrechtmäßig geschah / geschieht, können Sie statt "
            "der Löschung die Einschränkung der Datenverarbeitung verlangen.\n"
            "-  Wenn wir Ihre personenbezogenen Daten nicht mehr benötigen, Sie sie jedoch zur Ausübung, Verteidigung "
            "oder Geltendmachung von Rechtsansprüchen benötigen, haben Sie das Recht, statt der Löschung die "
            "Einschränkung der Verarbeitung Ihrer personenbezogenen Daten zu verlangen.\n"
            "-  Wenn Sie einen Widerspruch nach Art. 21 Abs. 1 DSGVO eingelegt haben, muss eine Abwägung zwischen "
            "Ihren und unseren Interessen vorgenommen werden. Solange noch nicht feststeht, wessen Interessen "
            "überwiegen, haben Sie das Recht, die Einschränkung der Verarbeitung Ihrer personenbezogenen Daten zu "
            "verlangen.")

st.markdown("Wenn Sie die Verarbeitung Ihrer personenbezogenen Daten eingeschränkt haben, dürfen diese Daten – von "
            "ihrer Speicherung abgesehen – nur mit Ihrer Einwilligung oder zur Geltendmachung, Ausübung oder "
            "Verteidigung von Rechtsansprüchen oder zum Schutz der Rechte einer anderen natürlichen oder juristischen "
            "Person oder aus Gründen eines wichtigen öffentlichen Interesses der Europäischen Union oder eines "
            "Mitgliedstaats verarbeitet werden.")

st.markdown("""---""")

st.header("Datenschutzbeauftragter")

st.markdown("Gesetzlich vorgeschriebener Datenschutzbeauftragter")
st.markdown("Wir haben für unsere Universität einen behördlichen Datenschutzbeauftragten bestellt:")
st.markdown("Dipl.-Kfm. Björn Voitel\n\n"
            "Stabsstelle Datenschutz & IT-Sicherheit\n\n"
            "Universität Osnabrück\n\n"
            "Nelson-Mandela-Straße 4\n\n"
            "49076 Osnabrück\n\n"
            "Telefon: +49 541 969-7880\n\n"
            "datenschutzbeauftragter@uni-osnabrueck.de")

st.markdown("""---""")

st.header("Datenerfassung auf unserer Website")

st.subheader("Cookies")

st.markdown("Die Internetseiten verwenden teilweise so genannte Cookies. Cookies richten auf Ihrem Rechner keinen "
            "Schaden an und enthalten keine Viren. Cookies dienen dazu, unser Angebot nutzerfreundlicher, effektiver "
            "und sicherer zu machen. Cookies sind kleine Textdateien, die auf Ihrem Rechner abgelegt werden und die "
            "Ihr Browser speichert.")
st.markdown("Die meisten der von uns verwendeten Cookies sind so genannte “Session-Cookies”. Sie werden nach Ende "
            "Ihres Besuchs automatisch gelöscht. Andere Cookies bleiben auf Ihrem Endgerät gespeichert bis Sie diese "
            "löschen. Diese Cookies ermöglichen es uns, Ihren Browser beim nächsten Besuch wiederzuerkennen.")
st.markdown("Sie können Ihren Browser so einstellen, dass Sie über das Setzen von Cookies informiert werden und "
            "Cookies nur im Einzelfall erlauben, die Annahme von Cookies für bestimmte Fälle oder generell "
            "ausschließen sowie das automatische Löschen der Cookies beim Schließen des Browsers aktivieren. "
            "Bei der Deaktivierung von Cookies kann die Funktionalität dieser Website eingeschränkt sein.")
st.markdown("Cookies, die zur Durchführung des elektronischen Kommunikationsvorgangs oder zur Bereitstellung "
            "bestimmter, von Ihnen erwünschter Funktionen (z.B. Warenkorbfunktion) erforderlich sind, werden auf "
            "Grundlage von Art. 6 Abs. 1 lit. f DSGVO gespeichert. Der Websitebetreiber hat ein berechtigtes Interesse "
            "an der Speicherung von Cookies zur technisch fehlerfreien und optimierten Bereitstellung seiner Dienste. "
            "Soweit andere Cookies (z.B. Cookies zur Analyse Ihres Surfverhaltens) gespeichert werden, werden diese in "
            "dieser Datenschutzerklärung gesondert behandelt.")

st.subheader("Server-Log-Dateien")

st.markdown("Der Provider der Seiten erhebt und speichert automatisch Informationen in so genannten "
            "Server-Log-Dateien, die Ihr Browser automatisch an uns übermittelt. Dies sind:")

st.markdown("-  Browsertyp und Browserversion\n"
            "-  verwendetes Betriebssystem\n"
            "-  Referrer URL\n"
            "-  Hostname des zugreifenden Rechners\n"
            "-  Uhrzeit der Serveranfrage\n"
            "-  IP-Adresse")

st.markdown("Eine Zusammenführung dieser Daten mit anderen Datenquellen wird nicht vorgenommen.")
st.markdown("Die Erfassung dieser Daten erfolgt auf Grundlage von Art. 6 Abs. 1 lit. f DSGVO. Der Websitebetreiber hat "
            "ein berechtigtes Interesse an der technisch fehlerfreien Darstellung und der Optimierung seiner Website – "
            "hierzu müssen die Server-Log-Files erfasst werden.")

