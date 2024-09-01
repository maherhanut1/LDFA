
class E():
    
    def __init__(self) -> None:
        self.__E = None
      
    def update_E(self, E):
        self.__E = E
    
    def get_E(self):
        return self.__E
    
global Ewrapper
Ewrapper = E()