from kivymd.app import MDApp
from kivymd.uix.label import MDLabel


class EmotionsApp(MDApp):
    def build(self):
        return MDLabel(text="Hello, World", halign="center")


EmotionsApp().run()