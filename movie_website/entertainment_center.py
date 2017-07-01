#encoding=utf-8
import media
import fresh_tomatoes

wukong = media.Movie("悟空传","http://www.truemovie.com/2016photo/WuKong-2.jpg",
						 "https://youtu.be/iNXew6zQXLQ")
gintama = media.Movie("銀魂","http://www.truemovie.com/2016photo/Gintama-1.jpg",
						"https://youtu.be/DLXlwxDXgVI")
jumanji = media.Movie("野蛮游戏:瘋狂叢林","http://www.truemovie.com/2016photo/jumanji-mit-dwayne-johnson-jack-black-karen-gillan-und-kevin-hart.jpg",
	                  "https://www.youtube.com/watch?v=lsdJA-Fca-I")
overdrive = media.Movie("盜速飛車","http://www.truemovie.com/2016photo/overdrive-mit-ana-de-armas-scott-eastwood-clemens-schick-und-freddie-thorp.jpg",
						"https://youtu.be/59p-YL9IUXg")

movies = [wukong, gintama, jumanji, overdrive]
fresh_tomatoes.open_movies_page(movies)