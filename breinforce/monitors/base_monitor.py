import os
from breinforce import views 


class Monitor:
    """Base monitor. Any recorder must subclass this recorder.
    """

    def __init__(self) -> None:
        self.history = []

    def add(self, fact) -> None:
        self.history.append(fact)

    def start(self, filename, mode="ascii") -> None:
        file_path = os.path.join("results", filename)
        os.makedirs("results", exist_ok=True)
        self.file = open(file_path, "w+")
        self.mode = mode

    def stop(self) -> None:
        string = None
        if self.mode == "ascii":
            string = views.AsciiView(self.history) 
        elif self.mode == "hands":
            string = views.HandsView(self.history)
        self.file.write(string)
