import java.io.*;
import java.util.*;
public class Main
{

    int INPUT_NEURONS = 7;
    int HIDDEN_NEURONS = 30;
    int OUTPUT_NEURONS = 5;
    int NUMBER_OF_LAYERS = 3;
    int TOTAL_DATA_SIZE = 1000000;
    int TRAINING_PICS = 700000;
    int TEST = 300000;

    NeuralNetwork nn;
    
    public static void main (String [] args) throws FileNotFoundException, OutOfMemoryError, IOException {
    	Main m = new Main();
    }

    public Main() throws FileNotFoundException, IOException, OutOfMemoryError{
        long startTime = System.currentTimeMillis();

        int sizes[] = {INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS};
        SigmoidNeuron hidden[] = new SigmoidNeuron[HIDDEN_NEURONS];
        SigmoidNeuron output[] = new SigmoidNeuron[OUTPUT_NEURONS];
        nn = new NeuralNetwork(NUMBER_OF_LAYERS, sizes, hidden, output);

        //Load training data
        Data d;
        Picture pics[] = new Picture[TOTAL_DATA_SIZE];
        System.out.println("Loading training data...");
        BufferedReader reader;
        File f;


        f = new File("data_compact.csv");
        reader = new BufferedReader(new FileReader(f));
        for (int p = 0; p < TOTAL_DATA_SIZE; p++){
            String answer = reader.readLine().replaceAll(" ", "");
            pics[p] = new Picture(new double[784], new int[10]);
            for (int y = 0; y < 10; y++){
                int index = answer.indexOf(",");
                String a = answer.substring(0,index);
                pics[p].output[y] = Integer.parseInt(a);
                answer = answer.substring(index + 1, answer.length());
            }

            String string = reader.readLine().replaceAll(" ", "");
            for (int x = 0; x < 784; x++){
                int index = string.indexOf(",");
                String pixel = string.substring(0,index);
                pics[p].pixels[x] = (double)(Integer.parseInt(pixel)) / 256;
                string = string.substring(index + 1, string.length());
            }

        }      
        d = new Data(pics, pics.length);
        reader.close();

        //Load testing data
        Data t;
        pics = new Picture[TEST];
        System.out.println("Loading testing data...");
        reader = new BufferedReader(new FileReader(f));
        for (int p = 0; p < TEST; p++){
            pics[p] = new Picture(new double[784], new int[10]);

            String answer = reader.readLine().replaceAll(" ", "");
            for (int y = 0; y < 10; y++){
                int index = answer.indexOf(",");
                String a = answer.substring(0,index);
                pics[p].output[y] = Integer.parseInt(a);
                answer = answer.substring(index + 1, answer.length());
            }

            String string = reader.readLine().replaceAll(" ", "");
            for (int x = 0; x < 784; x++){
                int index = string.indexOf(",");
                String pixel = string.substring(0,index);
                pics[p].pixels[x] = (double)(Integer.parseInt(pixel)) / 256;
                string = string.substring(index + 1, string.length());
            }
        }
        reader.close();
        t = new Data(pics, pics.length);

        randomizeWB();

        int EPOCHS = 20;
        int M_BATCH_SIZE = 10;
        int LEARNING_RATE = 3;

        System.out.println("Training / Testing network...");
        SGD(d, EPOCHS, M_BATCH_SIZE, LEARNING_RATE, t);

        long stopTime = System.currentTimeMillis();
        System.out.println("Elapsed time was " + (stopTime - startTime) / 1000 + " seconds.");
    }

    public void randomizeWB(){
        System.out.println("Assigning random values...");
        for (int i = 0; i < HIDDEN_NEURONS; i++){
            nn.hidden[i] = new SigmoidNeuron(0.0,new double[784], 0.0, new double[784]);
            nn.hidden[i].bias = Math.random()*2.0 - 1.0;
            nn.hidden[i].dBias = 0;
            for (int j = 0; j < INPUT_NEURONS; j++){
                nn.hidden[i].weights[j] = Math.random()*2.0 - 1.0;
                nn.hidden[i].dWeights[j] = 0;
            }
        }
        for (int i = 0; i < OUTPUT_NEURONS; i++){
            nn.output[i] = new SigmoidNeuron(0.0,new double[HIDDEN_NEURONS], 0.0, new double[HIDDEN_NEURONS]);
            nn.output[i].bias = Math.random()*2.0 - 1.0;
            nn.output[i].dBias = 0;
            for (int j = 0; j < HIDDEN_NEURONS; j++){
                nn.output[i].weights[j] = Math.random()*2.0 - 1.0;
                nn.output[i].dWeights[j] = 0;
            }
        }
    }

    /*public void loadWB() throws FileNotFoundException, IOException{
    BufferedReader reader;
    File f;

    //Output bias
    f = new File("./ob_ted");
    reader = new BufferedReader(new FileReader(f));
    for (int i = 0; i < OUTPUT_NEURONS; i++) {
    nn.output[i] = new SigmoidNeuron(0.0,new double[HIDDEN_NEURONS], 0.0, new double[HIDDEN_NEURONS]);
    String bias = reader.readLine();
    nn.output[i].bias = Double.parseDouble(bias);
    }
    reader.close();

    // Output weights  
    f = new File("./ow_ted");
    reader = new BufferedReader(new FileReader(f));
    for (int i = 0; i < OUTPUT_NEURONS; i++) {
    for (int j = 0; j < HIDDEN_NEURONS; j++) {
    String weight = reader.readLine();
    nn.output[i].weights[j] = Double.parseDouble(weight);
    }
    }
    reader.close();

    // Hidden bias 
    f = new File("./hb_ted");
    reader = new BufferedReader(new FileReader(f));
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
    nn.hidden[i] = new SigmoidNeuron(0.0,new double[784], 0.0, new double[784]);
    String bias = reader.readLine();
    nn.hidden[i].bias = Double.parseDouble(bias);
    }
    reader.close();

    // Hidden weights  
    f = new File("./hw_ted");
    reader = new BufferedReader(new FileReader(f));
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
    for (int j = 0; j < 784; j++) {
    String weight = reader.readLine();
    nn.hidden[i].weights[j] = Double.parseDouble(weight);
    }
    }
    reader.close();
    }

    public ArrayList<String> getFiles(String dir) throws FileNotFoundException, IOException{
    File folder = new File(dir);
    File[] listOfFiles = folder.listFiles();
    ArrayList<String> data = new ArrayList<String>();
    for (File file : listOfFiles) {
    BufferedReader reader = new BufferedReader(new FileReader(file));
    String s = reader.readLine();
    while(s != null){
    data.add(s.replaceAll(" ", ""));
    s = reader.readLine();
    }
    reader.close();
    }
    return data;
    }
     */

    public double[] feedForward(double [] a){
        double output[] = new double[OUTPUT_NEURONS];
        double sum;
        double temp[] = new double[HIDDEN_NEURONS];
        double temp2[] = new double[OUTPUT_NEURONS];

        for(int i = 0; i < HIDDEN_NEURONS; i++){
            sum = 0;
            for(int j = 0; j < INPUT_NEURONS; j++)
                sum += nn.hidden[i].weights[j] * a[j];
            temp[i] = sum + nn.hidden[i].bias;
        }
        temp = sigmoid(temp);


        for(int i = 0; i < OUTPUT_NEURONS; i++){
            sum = 0;
            for(int j = 0; j < HIDDEN_NEURONS; j++)
                sum += nn.output[i].weights[j] * temp[j];
            temp2[i] = sum + nn.output[i].bias;
        }
        output = sigmoid(temp2);
        /*
        for(int i = 0; i < OUTPUT_NEURONS; i++){
        sum = 0;
        for(int j = 0; j < HIDDEN_NEURONS; j++)
        sum += nn.output[i].weights[j] * temp[j];
        temp2[i] = sum + nn.output[i].bias;
        }
        output = sigmoid(temp2);
         */
        return output;
    }

    /*public double[] feedBackward(double input[]) {
    double temp[] = new double[HIDDEN_NEURONS];
    double output[] = new double[INPUT_NEURONS];
    for (int j = 0; j < HIDDEN_NEURONS; j++)
    temp[j] = 0.0;
    for (int i = 0; i < HIDDEN_NEURONS; i++){
    for (int j = 0; j < OUTPUT_NEURONS; j++){
    temp[i] += nn.output[j].weights[i] * input[j];
    }
    }
    for (int i = 0; i < HIDDEN_NEURONS; i++){
    for (int j = 0; j < 784; j++){
    output[j] += nn.hidden[i].weights[j] * temp[i];
    }
    }
    double min = output[0];
    double max = output[0];
    for (int j = 0; j < 784; j++){
    if (output[j] > max)
    max = output[j];
    if (output[j] < min)
    min = output[j];
    }
    for (int j = 0; j < 784; j++){
    output[j] = Math.abs(output[j]) / (max - min);
    }
    return output;
    }
     */

    public void SGD(Data trainingData, int epochs, int miniBatchSize, int eta, Data testData){
        MiniBatch batch = new MiniBatch(new Picture[miniBatchSize], 0);

        for(int e = 0; e < epochs; e++){
            double numBatchs = trainingData.nPics / miniBatchSize;

            for(int i = 0; i < numBatchs; i++){
                batch.nPics = miniBatchSize;
                for(int j = 0; j < miniBatchSize; j++){
                    int idx = i * miniBatchSize + j;
                    batch.pictures[j] = trainingData.data[idx];
                }
                if(batch.pictures[0] != null && batch.pictures[batch.pictures.length - 1] != null)
                    updateMiniBatch(batch, eta);
            }

            if(testData != null){
                int result = evaluate(testData);
                System.out.println("Epoch " + (e + 1) + " : " + result + " / " + testData.nPics);
            }
            else
                System.out.println("Epoch " + (e + 1) + " complete.");
        }
    }

    void updateMiniBatch(MiniBatch batch, int eta){
        int n = batch.nPics;

        for(int i = 0; i < n; i++){
            backprop(batch.pictures[i]);
            for(int j = 0; j < OUTPUT_NEURONS; j++){
                double db = nn.output[j].dBias;
                nn.output[j].bias -= db * eta / n;
                for(int k = 0; k < HIDDEN_NEURONS; k++){
                    double dw = nn.output[j].dWeights[k];
                    nn.output[j].weights[k] -= dw * eta / n;
                }
            }

            for(int j = 0; j < HIDDEN_NEURONS; j++){
                double db = nn.hidden[j].dBias;
                for(int k = 0; k < INPUT_NEURONS; k++){
                    double dw = nn.hidden[j].dWeights[k];
                    nn.hidden[j].weights[k] -= dw * eta / n;
                }
                nn.hidden[j].bias -= db * eta / n;
            }
        }
    }

    public void backprop(Picture p){
        double sum = 0;
        double zh[] = new double[HIDDEN_NEURONS];
        double ah[] = new double[HIDDEN_NEURONS];
        double zo[] = new double[OUTPUT_NEURONS];
        double ao[] = new double[OUTPUT_NEURONS];

        for(int i = 0; i < HIDDEN_NEURONS; i++){
            sum = 0;
            for(int j = 0; j < INPUT_NEURONS; j++)
                sum += nn.hidden[i].weights[j] * p.pixels[j];
            zh[i] = sum + nn.hidden[i].bias;
        }
        ah = sigmoid(zh);

        for(int i = 0; i < OUTPUT_NEURONS; i++){
            sum = 0;
            for(int j = 0; j < HIDDEN_NEURONS; j++)
                sum += nn.output[i].weights[j] * ah[j];
            zo[i] = sum + nn.output[i].bias;
        }
        ao = sigmoid(zo);

        double dHidden[] = new double[HIDDEN_NEURONS];
        double dOutput[] = new double[OUTPUT_NEURONS];

        double c_d[] = new double[OUTPUT_NEURONS];
        double s_p[] = new double[OUTPUT_NEURONS];
        int y[] = p.output;
        c_d = cost_derivative(zo, y);
        s_p = sigmoid_prime(zo);
        for(int i = 0; i < OUTPUT_NEURONS; i++){
            dOutput[i] = c_d[i] * s_p[i];
            for(int j = 0;  j < HIDDEN_NEURONS; j++)
                nn.output[i].dWeights[j] = ah[j] * dOutput[i];
            nn.output[i].dBias = dOutput[i];
        }

        double s_ph[] = new double[HIDDEN_NEURONS];
        s_ph = sigmoid_prime(zh);
        for(int i = 0; i < HIDDEN_NEURONS; i++){
            double G = 0;
            for(int j = 0; j < OUTPUT_NEURONS; j++)
                G += dOutput[j] * nn.output[j].weights[i];
            dHidden[i] = G * s_ph[i];

            for(int j = 0; j < INPUT_NEURONS; j++)
                nn.hidden[i].dWeights[j] = p.pixels[j]*dHidden[i];
            nn.hidden[i].dBias = dHidden[i];
        }
    }

    public int evaluate(Data testData){
        int n = (int)(testData.nPics);
        int sum = 0;
        for(int i = 0; i < n; i++)
        {
            Picture p = testData.data[i];
            double output[] = new double[OUTPUT_NEURONS];
            for(int j = 0; j < OUTPUT_NEURONS; j++)
                output[j] = 0.0;
            output = feedForward(p.pixels);
            int r = maxd(output);
            int why = maxi(p.output);
            if (r == why){
                sum++;
            }
            else{
                store(p, output);
            }
        }
        return sum;
    }

    int index = 0;
    Data d = new Data(new Picture[169], 0);
    ArrayList<double[]> outputs = new ArrayList<double[]>();

    public void store(Picture p, double[] output){
        if(this.d.nPics < this.d.data.length){
            this.d.data[this.d.nPics] = p;
            this.d.nPics++;
            outputs.add(output);
        }
    }

    public int maxd(double[] input){
        double m = input[0];
        int r = 0;
        for(int i = 0; i < input.length; i++)
            if(input[i] > m){
                m = input[i];
                r = i;
            }
        return r;
    }

    public int maxi(int[] input){
        int m = input[0];
        int r = 0;
        for(int i = 0; i < input.length; i++)
            if(input[i] > m){
                m = input[i];
                r = i;
            }
        return r;
    }

    public double[] cost_derivative(double[] output_a, int[] y){
        double[] result = new double[output_a.length];
        for(int i = 0; i < output_a.length; i++)
            result[i] = output_a[i] - y[i];
        return result;
    }

    public double[] sigmoid(double[] z){
        double[] a = new double[z.length];
        for(int i = 0; i < z.length; i++)
            a[i] = 1.0 / (1.0 + Math.exp(-1*z[i]));
        return a;
    }

    public double[] sigmoid_prime(double[] z){
        double s[] = new double[HIDDEN_NEURONS];
        double result[] = new double[HIDDEN_NEURONS];
        s = sigmoid(z);
        for(int i = 0; i < z.length; i++)
            result[i] = s[i] * (1-s[i]);
        return result;  
    }

    public class SigmoidNeuron{
        double bias;
        double weights[];
        double dBias;
        double dWeights[];
        public SigmoidNeuron(double bias, double weights[],
        double dBias, double dWeights[]){
            this.bias = bias;
            this.weights = weights;
            this.dBias = dBias;
            this.dWeights = dWeights;
        }
    }
    public class NeuralNetwork{
        int nLayers;
        int sizes[];
        SigmoidNeuron hidden[];
        SigmoidNeuron output[];
        public NeuralNetwork(int nLayers, int sizes[], 
        SigmoidNeuron hidden[], SigmoidNeuron output[]){
            this.nLayers = nLayers;
            this.sizes = sizes;
            this.hidden = hidden;
            this.output = output;
        }
    }
    public class MiniBatch{
        Picture pictures[];
        int nPics;
        public MiniBatch(Picture pictures[], int nPics){
            this.pictures = pictures;
            this.nPics = nPics;
        }
    }
}
