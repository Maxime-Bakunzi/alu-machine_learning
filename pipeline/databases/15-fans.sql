-- Ranks counrty origins of bands by number of fans
SELECT orign, SUM(fans) as nb_fans
FROM metalbands
GROUP BY orign
ORDER BY nb_fans DESC;