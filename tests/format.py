# %%
Companies = """
xAI
Chai Discovery
Lila sciences
Perplexity
Zoox
Mercor
Contextual AI
Datadog
Vatic Labs
Calico
Luma Labs
LMArena
Sesame AI
Virtu
Cohere
DESRES
HRT
Google
Netflix
Meta
Apple
Haize Labs
Gray Swan AI
DE Shaw Group
DESRES
HRT
Datadog
Palantir
Point72
Reddit
ElevenLabs
Perplexity
Magic
xAI
ARC
Calico
Waymo
Together
Stripe
Zoox
Astera
Nuro
Databricks
Cubist
IMC
Scale AI
Hinge
"""
sCompanies = set()
for company in Companies.split("\n"):
    if company:
        sCompanies.add(company.lower())
print(sorted(list(sCompanies)))
# %%
B_TIER = """
Spotify
Ramp
Figma
Vercel
Notion
AirBnb
DRW
Mistral AI
Suno
Voleon Group
Together AI
Glean
Windsurf
"""