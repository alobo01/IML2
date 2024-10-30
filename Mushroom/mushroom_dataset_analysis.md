# Mushroom Poison Classification

## Introduction
Mushroom foraging has gained popularity in recent years, yet it comes with the significant risk of accidental poisoning. Identifying which mushrooms are edible and which are toxic is crucial for both enthusiasts and researchers. Machine learning models can aid in mushroom classification based on physical characteristics, helping predict the likelihood of toxicity based on specific features. Research shows that some species possess subtle indicators that suggest toxicity, such as odor and cap color, which can be used in classification tasks to guide safer foraging practices ([Largent et al., 1977](https://www.jstor.org/stable/3799781)).

This document describes the attributes of a dataset used to predict the edibility of mushrooms, helping researchers and models recognize poisonous species based on observable characteristics.

## Dataset Attributes
Below is a breakdown of each attribute in the dataset and the values it may take. These attributes capture physical characteristics or environmental indicators that correlate with mushroom toxicity.

1. **cap-shape**: Refers to the shape of the mushroom's cap.
   - Values: bell (`b`), conical (`c`), convex (`x`), flat (`f`), knobbed (`k`), sunken (`s`)

2. **cap-surface**: Describes the surface texture of the cap.
   - Values: fibrous (`f`), grooves (`g`), scaly (`y`), smooth (`s`)

3. **cap-color**: Indicates the cap color, which may be associated with toxicity.
   - Values: brown (`n`), buff (`b`), cinnamon (`c`), gray (`g`), green (`r`), pink (`p`), purple (`u`), red (`e`), white (`w`), yellow (`y`)

4. **bruises?**: Shows if the mushroom cap bruises when damaged.
   - Values: bruises (`t`), no (`f`)

5. **odor**: Odor can often indicate toxicity in mushrooms.
   - Values: almond (`a`), anise (`l`), creosote (`c`), fishy (`y`), foul (`f`), musty (`m`), none (`n`), pungent (`p`), spicy (`s`)

6. **gill-attachment**: Describes the attachment of gills to the mushroom stem.
   - Values: attached (`a`), descending (`d`), free (`f`), notched (`n`)

7. **gill-spacing**: Describes the spacing of the mushroom’s gills.
   - Values: close (`c`), crowded (`w`), distant (`d`)

8. **gill-size**: Refers to the size of the gills.
   - Values: broad (`b`), narrow (`n`)

9. **gill-color**: Indicates the gill color, which may relate to toxicity.
   - Values: black (`k`), brown (`n`), buff (`b`), chocolate (`h`), gray (`g`), green (`r`), orange (`o`), pink (`p`), purple (`u`), red (`e`), white (`w`), yellow (`y`)

10. **stalk-shape**: Describes the shape of the mushroom stalk.
    - Values: enlarging (`e`), tapering (`t`)

11. **stalk-root**: Describes the root type of the stalk.
    - Values: bulbous (`b`), club (`c`), cup (`u`), equal (`e`), rhizomorphs (`z`), rooted (`r`), missing (`?`)

12. **stalk-surface-above-ring**: Surface texture above the ring on the stalk.
    - Values: fibrous (`f`), scaly (`y`), silky (`k`), smooth (`s`)

13. **stalk-surface-below-ring**: Surface texture below the ring on the stalk.
    - Values: fibrous (`f`), scaly (`y`), silky (`k`), smooth (`s`)

14. **stalk-color-above-ring**: Color of the stalk above the ring.
    - Values: brown (`n`), buff (`b`), cinnamon (`c`), gray (`g`), orange (`o`), pink (`p`), red (`e`), white (`w`), yellow (`y`)

15. **stalk-color-below-ring**: Color of the stalk below the ring.
    - Values: brown (`n`), buff (`b`), cinnamon (`c`), gray (`g`), orange (`o`), pink (`p`), red (`e`), white (`w`), yellow (`y`)

16. **veil-type**: Type of veil covering the mushroom.
    - Values: partial (`p`), universal (`u`)

17. **veil-color**: Color of the veil.
    - Values: brown (`n`), orange (`o`), white (`w`), yellow (`y`)

18. **ring-number**: Number of rings on the mushroom stalk.
    - Values: none (`n`), one (`o`), two (`t`)

19. **ring-type**: Type of ring on the mushroom stalk.
    - Values: cobwebby (`c`), evanescent (`e`), flaring (`f`), large (`l`), none (`n`), pendant (`p`), sheathing (`s`), zone (`z`)

20. **spore-print-color**: Color of the spore print.
    - Values: black (`k`), brown (`n`), buff (`b`), chocolate (`h`), green (`r`), orange (`o`), purple (`u`), white (`w`), yellow (`y`)

21. **population**: Population density where the mushroom grows.
    - Values: abundant (`a`), clustered (`c`), numerous (`n`), scattered (`s`), several (`v`), solitary (`y`)

22. **habitat**: Type of habitat where the mushroom is found.
    - Values: grasses (`g`), leaves (`l`), meadows (`m`), paths (`p`), urban (`u`), waste (`w`), woods (`d`)

23. **class**: Classification of the mushroom as edible or poisonous.
    - Values: edible (`e`), poisonous (`p`)