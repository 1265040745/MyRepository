#encoding=utf-8

class Movie(object):
	"""A description of the movie
		Invoking the current class 
		to create a movie is an instance object
	"""
	def __init__(self,  movie_title, poster_image,
				 trailer_youtube):
	"""Method description
		These parameters are properties of the movie
		Movie properties: title, poster_image_url, 
		trailer_youtube_url
	"""
		self.title = movie_title
		self.poster_image_url = poster_image
		self.trailer_youtube_url = trailer_youtube

