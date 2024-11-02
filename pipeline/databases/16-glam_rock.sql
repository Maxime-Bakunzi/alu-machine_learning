-- Lists all Glam rock bands, ranked by their longevity
SELECT
    band_name,
    IFNULL(2020 - formed, 0) - IFNULL(IFNULL(split, 2020) - formed, 0) + IFNULL(2020 - formed, 0) as lifespan
FROM metal_bands
WHERE FIND_IN_SET('Glam rock', IFNULL(style, '')) > 0
ORDER BY lifespan DESC;