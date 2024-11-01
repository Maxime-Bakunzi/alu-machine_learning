-- Displays the average temperature (Fahrenheit) by city ordered by temerature (descending)
SELECT city, AVG(value) AS avg_temp
FROM temeratures
GROUP BY city
ORDER BY avg_temp DESC;