import numpy as np
import matplotlib.pyplot as plt

def DFT(x): #ndarray
    """
    Compute the discrete Fourier Transform of the 1D array x
    :param x: (array)
    """
    # N = number of samples
    # n = current sample
    # xn = value of the signal at time n
    # k = current frequency (0 Hz to N-1 Hz)
    # Xk = Result of the DFT (amplitude and phase)

    N = x.size
    # print("size: ", N)
    
    n = np.arange(N) # Stop is x.size, start is 0, and step is 1
    # print("np.arange(N): ", n)
    
    k = n.reshape((N, 1))
    # print("n.reshape((N, 1)): ", k)

    e = np.exp(-2j * np.pi * k * n / N)
    print("e: ", e)
    print '--------------'
    print("e.size: ", e.size)

    return np.dot(e, x)

# -----------------------------------------------

def plotFFT():
    print("Plot FFT start")
    t = np.linspace(0, 1, 1300) #from 0 seconds, to 1 second, with 1300 sampling points
    # 40hz + 50% of 90hz - we are dampening the 90hz by half
    s = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)

    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    
    # print("time: " , t)
    # print("--------")
    # print("value at time [x]: " , s)

    # plt.plot(t, s)
    # plt.show()

    fft = np.fft.fft(s)

    # for i in range(2):
    #     print("Value at index {}:\t{}".format(i, fft[i + 1]), "\nValue at index {}:\t{}".format(fft.size -1 - i, fft[-1 - i]))


    fft = np.fft.fft(s)
    T = t[1] - t[0]  # sampling interval 
    N = s.size

    # 1/T = frequency
    f = np.linspace(0, 1 / T, N)

    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")

    # print("f[:N // 2]: ",f[:N // 3])
    print("1 / 1300 : ", f[:N//2] )

    plt.bar(f[:N //2],                 #X - frequencies
        np.abs(fft)[:N // 2] * 1 / N    #Y - 1 / N is a normalization factor
        , width=5)                      #width of the bar
    # plt.show()


def main():
    # myArray = np.linspace(0, 0.5, 20000)
    # result = DFT(myArray)
    # print(result)
    plotFFT()

if __name__ == "__main__":
    main()