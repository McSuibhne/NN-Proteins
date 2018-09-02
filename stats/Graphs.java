import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Scanner;

//Parsing scripts to convert neural network output messages into easily graph-able lists of error values and percentages
public class Graphs {
public static boolean REVERSE = false;
    public static void main(String[] args) {
        findErrors();
        findPercents();
    }

    static void findErrors() {
        ArrayList<String> readnames = new ArrayList<String>();
        ArrayList<String> writenames = new ArrayList<String>();

        readnames.add("5fold0lrate000001.txt");
        writenames.add("5errors000001.txt");
        for (int i = 0; i < 4; i++) {
            try {
                String writefilename = writenames.get(i);
                String readfilename = readnames.get(i);

                Scanner scanner = new Scanner(new FileReader(readfilename));
                BufferedWriter writer = new BufferedWriter(new FileWriter(writefilename));

                while (scanner.hasNextLine()) {
                    String line = scanner.nextLine();
                    if (line.contains("Error=")) {
                        String temp = line.substring((line.length() - 8));
                        if(temp.contains("=")){
                            temp = temp.split(" ")[1];
                        }
                        writer.write(temp + "\n");
                    }
                }
                writer.close();
                scanner.close();
            } catch (FileNotFoundException fnfe) {
                System.out.println("File Not Found");
            } catch (java.io.IOException uee) {
                System.out.println("Could not write to file");
            }
        }
    }

    static void findPercents() {
        ArrayList<String> readnames = new ArrayList<String>();
        ArrayList<String> writenames = new ArrayList<String>();

        readnames.add("13fold4loga.txt");
        writenames.add("13percents.txt");
        for (int i = 0; i < 1; i++) {
            try {
                String writefilename = writenames.get(i);
                String readfilename = readnames.get(i);
                String train_line = "";
                boolean train_found = false;
                Scanner scanner = new Scanner(new FileReader(readfilename));
                BufferedWriter writer = new BufferedWriter(new FileWriter(writefilename));

                while (scanner.hasNextLine()) {
                    String line = scanner.nextLine();
                    if (line.contains("NErrors=")) {
                        String figure = line.substring((line.length() - 7));
                        if (REVERSE) {
                            Double temp = (100 - Double.parseDouble(figure));
                            figure = temp.toString();
                        }
                        if (train_found) {
                            writer.write(train_line + figure + "\n");
                            train_line = "";
                            train_found = false;
                        } else {
                            train_line = figure + "\t";
                            train_found = true;
                        }
                    }
                }
                writer.close();
                scanner.close();
            } catch (FileNotFoundException fnfe) {
                System.out.println("File Not Found");
            } catch (java.io.IOException uee) {
                System.out.println("Could not write to file");
            }
        }
    }
}