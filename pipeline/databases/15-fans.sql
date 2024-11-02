-- Ranks counrty origins of bands by number of fans
SELECT orign, SUM(fans) as nb_fans
FROM ab2979f058de215f0f2ae5b052739e76d3c02ac5
GROUP BY orign
ORDER BY nb_fans DESC;