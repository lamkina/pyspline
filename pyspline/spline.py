class Spline(object):
    def __init__(self):
        self._name = ""
        self._tag = ""
        self._rational = False

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def tag(self) -> str:
        return self._tag

    @tag.setter
    def tag(self, tag: str) -> None:
        self._tag = tag
