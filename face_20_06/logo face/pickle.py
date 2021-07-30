import pickle


class Person:
  def __init__(self, name, age):
      self.name = name
      self.age = age

  def info(self):
      print "Name: " + self.name
      print "Age: " + str(self.age)


p1 = Person("lkintheend", 21)

with open("pickle.txt", "wb") as f:
  pickle.dump(p1, f)
