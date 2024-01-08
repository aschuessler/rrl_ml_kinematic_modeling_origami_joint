import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class FigureMaker:
    """
    Class to generate figures for force feedback response
    """
    def __init__(self, file_names):
        # get csv files
        self.roll_50 = f"experiments/feedback_response/{file_names[0]}"
        self.roll_100 = f"experiments/feedback_response/{file_names[1]}"
        self.roll_150 = f"experiments/feedback_response/{file_names[2]}"
        self.pitch_50 = f"experiments/feedback_response/{file_names[3]}"
        self.pitch_100 = f"experiments/feedback_response/{file_names[4]}"
        self.pitch_150 = f"experiments/feedback_response/{file_names[5]}"
        self.df_roll_50 = pd.read_excel(self.roll_50).iloc[15000:20000:20, :].reset_index()
        self.df_roll_100 = pd.read_excel(self.roll_100).iloc[19300:24300:20, :].reset_index()
        self.df_roll_150 = pd.read_excel(self.roll_150).iloc[11670:16670:20, :].reset_index()
        self.df_pitch_50 = pd.read_excel(self.pitch_50).iloc[6000:11000:20, :].reset_index()
        self.df_pitch_100 = pd.read_excel(self.pitch_100).iloc[12700:17700:20, :].reset_index()
        self.df_pitch_150 = pd.read_excel(self.pitch_150).iloc[10000:15000:20, :].reset_index()

    def plot_force(self):
        # Reduce sample frequency to
        plt.figure("Roll", figsize=(8, 6))
        plt.plot(self.df_roll_150.index / 100, -self.df_roll_150.iloc[:, 3] + self.df_roll_150.iloc[0, 3], c="black", linestyle="solid", label="150 mA")
        plt.plot(self.df_roll_100.index / 100, -self.df_roll_100.iloc[:, 3] + self.df_roll_100.iloc[0, 3], c="black", linestyle="dashed", label="100 mA")
        plt.plot(self.df_roll_50.index / 100, -self.df_roll_50.iloc[:, 3] + self.df_roll_50.iloc[0, 3], c="black", linestyle="dotted", label="50 mA")

        print("- Roll -")
        print(f"Maximum force for 150: {-self.df_roll_150.iloc[:, 3].min()}")
        print(f"Maximum force for 100: {-self.df_roll_100.iloc[:, 3].min()}")
        print(f"Maximum force for 50: {-self.df_roll_50.iloc[:, 3].min()}")

        plt.xlabel('Time (s)', fontsize=18)
        plt.xticks(fontsize=14)
        plt.ylabel('Normal Force (N)', fontsize=18)
        plt.yticks(fontsize=14)
        plt.ylim([-0.1, 2.0])
        plt.legend(loc="upper left", fontsize=14)  # .set_title("Current limit")

        plt.figure("Pitch", figsize=(8, 6))
        plt.plot(self.df_pitch_150.index / 100, -self.df_pitch_150.iloc[:, 3] + self.df_pitch_150.iloc[0, 3], c="black", linestyle="solid", label="150 mA")
        plt.plot(self.df_pitch_100.index / 100, -self.df_pitch_100.iloc[:, 3] + self.df_pitch_100.iloc[0, 3], c="black", linestyle="dashed", label="100 mA")
        plt.plot(self.df_pitch_50.index / 100, -self.df_pitch_50.iloc[:, 3] + self.df_pitch_50.iloc[0, 3], c="black", linestyle="dotted", label="50 mA")

        print("- Pitch -")
        print(f"Maximum force for 150: {-self.df_pitch_150.iloc[:, 3].min()}")
        print(f"Maximum force for 100: {-self.df_pitch_100.iloc[:, 3].min()}")
        print(f"Maximum force for 50: {-self.df_pitch_50.iloc[:, 3].min()}")

        plt.xlabel('Time (s)', fontsize=18)
        plt.xticks(fontsize=14)
        plt.ylabel('Normal Force (N)', fontsize=18)
        plt.yticks(fontsize=14)
        plt.ylim([-0.1, 2.0])
        plt.legend(fontsize=14, loc="upper left")

        plt.show()


if __name__ == "__main__":
    excel_files = ["231116_roll_50_final.xlsx",
                   "231116_roll_100_final.xlsx",
                   "231116_roll_150_final.xlsx",
                   "231122_pitch_50_final.xlsx",
                   "231122_pitch_100_final.xlsx",
                   "231122_pitch_150_final.xlsx"]

    fig_maker = FigureMaker(excel_files)
    fig_maker.plot_force()
