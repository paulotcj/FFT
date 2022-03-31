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
    print ("--------------")
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

    plt.plot(t, s)
    plt.show()

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
    plt.show()

def plotFFT3hz():

    t = np.linspace(0, 2, 2000) #from 0 seconds, to 2 seconds, with 2000 sampling points
    # t = [0.001 for i in range(2000)] 
    # i = 0

    # j = 0
    # while j < 2000:
    #     i += 0.001
    #     t[j] = i
    #     j+=1


    s = (np.sin(3 * 2 * np.pi * t - 1.571) +1)  #+ (np.sin(5 * 2 * np.pi * t - 1.571) + 1) # sin wave where we derive the values from the samples

    #----------------------------
    # plt.xlabel("Time [s]")
    # plt.ylabel("Amplitude")
    # plt.plot(t, s)
    # plt.show()
    #----------------------------

    fft = np.fft.fft(s)

    T = t[1] - t[0]  # sampling STEP interval 

    N = s.size

    oneDivByT = 1 / T
    # oneDivByT = 999.5


    f = np.linspace(0, oneDivByT, N) #start at zero, goes up to 999.5, and takes 2000 steps

    xfreq = f[0:20] #takes the 20 first elements of F

    npabs_fft_ = np.abs(fft)

    yAmpli = npabs_fft_[0:20] * 1 / N 

    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")

    plt.bar(xfreq,                 #X - frequencies
        yAmpli    #Y - 1 / N is a normalization factor
        , width=1)                      #width of the bar
    plt.show()    

    varbusydebug =2


def main():
    # myArray = np.linspace(0, 0.5, 20000)
    # result = DFT(myArray)
    # print(result)
    # plotFFT()
    plotFFT3hz()

if __name__ == "__main__":
    main()