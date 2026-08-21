# Before you run this at scale — read this

This scraper collects publicly visible property listings from
fincaraiz.com.co. Running it is **your** decision as the operator, and the
legal responsibility is yours, not the tool's. Please work through this page
before any large run.

## 1. Terms of use

Read the site's *Términos y Condiciones* and *Política de Privacidad* at
<https://www.fincaraiz.com.co>. Automated collection may be restricted or
prohibited by those terms regardless of what `robots.txt` permits — the two are
separate things, and `robots.txt` permission is **not** legal permission.

If you need the data for anything beyond private academic study, contact
FincaRaíz and ask. Many portals will provide a data export or an API licence,
which is faster, more reliable and unambiguous.

## 2. Colombian data protection law

**Ley 1581 de 2012** (Régimen General de Protección de Datos Personales) and
its implementing **Decreto 1377 de 2013** govern the processing of personal
data in Colombia. Relevant points for this project:

- *Datos personales* means any information linked to an identifiable natural
  person. A named private individual selling a flat, with a phone number, is
  personal data. A registered agency's business contact details generally are
  not, but the line is not always crisp.
- Processing personal data normally requires **prior, express and informed
  consent** from the data subject. Scraping does not obtain consent.
- Data subjects hold rights of access, correction and deletion (*habeas
  data*), which you must be able to honour if you hold their data.
- Databases holding personal data may require registration with the
  **Registro Nacional de Bases de Datos (RNBD)** at the Superintendencia de
  Industria y Comercio.

**What this scraper does about it.** By design it never writes contact
details to disk. `owner.masked_phone`, `owner.whatsapp_phone`,
`owner.has_whatsapp` and `owner.subsidiaries` are dropped during parsing, and
when a listing is published by a private individual (`owner.particular` is
true) the agency name and id are nulled too. Street addresses are recorded
only where the site itself chooses to display them (`showAddress`).

This reduces the risk. **It does not make the dataset automatically lawful to
hold, publish or share.** In particular, do not republish the raw dataset, and
do not attempt to re-identify sellers by joining it against other sources.

## 3. Rate and etiquette

Defaults are deliberately conservative: 4 concurrent requests and a 1.5–3.0 s
jittered delay. `robots.txt` specifies **no** `Crawl-delay`, so this pacing is
our own choice rather than a site directive — please do not raise it just
because nothing stops you. A full arriendo/Bogotá run is on the order of an
hour; that is the polite cost of the data.

The scraper stops rather than evades. If it detects a 403, a captcha or a
Cloudflare interstitial it raises `BlockedError` and halts with a message.
**Do not add captcha solving, header spoofing beyond an ordinary browser
identity, or IP rotation to get around a block.** A block is the site telling
you to stop, and working around it moves you from "reading a public page" into
territory that is much harder to defend.

## 4. Sensible practice

- Keep `out/` out of version control and off shared drives.
- Prefer aggregates in anything you publish. A model fitted on the data is
  fine; a copy of the listings table is a republication.
- Delete raw caches when the project ends.
- Re-read this page if you change the target city, scale up, or start
  scheduling recurring runs.

*This is engineering guidance written to help you ask the right questions, not
legal advice. If the project moves beyond coursework, get a lawyer's read.*
