"""
A small hand-written dataset so the project runs without downloading anything.

Each item is (article_text, label), where label 0 = real news and 1 = fake news.

This is only enough to test the pipeline. A real model needs thousands of
articles. For that, use the Kaggle "Fake and Real News" dataset (see README.md).
"""

SAMPLE_DATA = [
    # Real news (label 0)
    ("The central bank announced a quarter-point interest rate cut on Wednesday, "
     "citing slowing manufacturing output and easing inflation figures.", 0),
    ("Researchers published a peer-reviewed study showing that the new vaccine "
     "reduced hospitalizations by 40 percent in a clinical trial of 12,000 adults.", 0),
    ("The city council approved a budget that increases funding for public transit "
     "and road repairs, according to a statement released Tuesday.", 0),
    ("Officials confirmed that the storm caused flooding in several coastal towns, "
     "and emergency crews are working to restore power to affected residents.", 0),
    ("The technology company reported quarterly earnings above analyst expectations, "
     "driven by growth in its cloud computing division.", 0),
    ("A national survey found that unemployment fell to its lowest level in three years, "
     "according to figures released by the labor department.", 0),
    ("The football club signed a five-year contract extension with its head coach "
     "following a successful season, the club announced on Monday.", 0),
    ("Lawmakers debated the proposed climate bill for six hours before sending it "
     "to committee for further review and public comment.", 0),
    ("The museum opened a new exhibition featuring works from the early twentieth "
     "century, with tickets available online starting this weekend.", 0),
    ("Health ministry data shows the number of registered schools in the district "
     "rose by eight percent over the past academic year.", 0),
    ("The airline said it will add twelve new routes next summer, expanding service "
     "to regional airports in the southern part of the country.", 0),
    ("Scientists at the observatory reported the discovery of a previously unknown "
     "comet, which will pass near Earth in early spring.", 0),
    # Fake news (label 1)
    ("BREAKING: Secret documents reveal the government is hiding alien technology "
     "in underground bunkers. Share before they delete this!", 1),
    ("Doctors don't want you to know this one simple trick that cures cancer in "
     "three days. Big pharma is covering it up, wake up people!", 1),
    ("Celebrity shocker: famous actor secretly replaced by a clone, insiders confirm. "
     "The mainstream media refuses to report the truth.", 1),
    ("You won't believe what this politician said in private. Leaked audio proves "
     "the whole election was rigged. Spread this everywhere now!", 1),
    ("Drinking bleach mixed with lemon juice boosts immunity by 500 percent, "
     "according to a miracle report the hospitals are hiding from you.", 1),
    ("Five-year-old discovers the moon is actually a hologram, scientists panic "
     "and refuse to comment. Full video shows the truth they tried to hide.", 1),
    ("URGENT: Microchips are being secretly added to grocery store products to track "
     "citizens. Forward to everyone you know before it is too late!", 1),
    ("Shocking proof that the ocean will swallow the coastline next week, a hidden "
     "NASA memo confirms what the elites have planned all along.", 1),
    ("Miracle weight loss pill melts fat overnight with zero effort, doctors furious "
     "that this secret has been kept from the public for decades.", 1),
    ("Anonymous insider exposes plan to ban all cash by Friday. Banks are already "
     "moving money offshore. Take action now before it is too late!", 1),
    ("The president was secretly replaced by a body double, eyewitnesses claim, "
     "and the government is paying people to stay silent about it.", 1),
    ("Viral video proves that the entire moon landing was filmed in a studio, "
     "experts who worked on it are finally speaking out. Share the truth!", 1),
]
