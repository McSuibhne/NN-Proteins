import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.*;

//Generates datasets of protein contact partners and encodes them for input into the neural network
public class ProteinParser {

    static String BASE_NAME = "pdb.11Dec2014.sorted_red25n.adataset";


    static ArrayList<Protein> proteins = new ArrayList<>();

    static HashMap<String, String> acid_codes = new HashMap<>();

    public static void main(String[] args) {
        setAcidCodes();
        generateSets(5);
    }

    public static void setAcidCodes() {
        acid_codes.put("A", "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("C", "0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("D", "0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("E", "0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("F", "0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("G", "0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("H", "0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("I", "0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("K", "0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("L", "0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("M", "0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("N", "0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0");
        acid_codes.put("P", "0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0");
        acid_codes.put("Q", "0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0");
        acid_codes.put("R", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0");
        acid_codes.put("S", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0");
        acid_codes.put("T", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0");
        acid_codes.put("V", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0");
        acid_codes.put("W", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0");
        acid_codes.put("Y", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0");
        acid_codes.put("B", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1");
        acid_codes.put("J", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1");
        acid_codes.put("O", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1");
        acid_codes.put("U", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1");
        acid_codes.put("X", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1");
        acid_codes.put("Z", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1");

    }


    static void buildProteins(String filename) {
        try {
            int line_count = 1;
            Scanner scanner = new Scanner(new FileReader(filename));

            while (scanner.hasNextLine()) {
                if (line_count % 16 == 1) {
                    String name = scanner.nextLine();
                    int length = Integer.parseInt(scanner.nextLine());
                    String[] primary_structure = scanner.nextLine().split("\t");
                    String[] secondary_structure = scanner.nextLine().split("\t");
                    String[] partner_indexes = scanner.nextLine().split("\t");

                    Protein protein = new Protein(name, length, primary_structure, secondary_structure, partner_indexes);
                    proteins.add(protein);
                    line_count += 5;
                } else {
                    scanner.nextLine();
                    line_count++;
                }
            }
        } catch (FileNotFoundException fnfe) {
            System.out.println("File Not Found");
        }
    }

    static ArrayList<ContactPartners> findSuccesses(int window_size) {
        ArrayList<ContactPartners> partners = new ArrayList<>();

        for (int i = 0; i < proteins.size(); i++) {
            partners.addAll(proteins.get(i).findPartners(window_size));
        }
        return partners;
    }

    static ArrayList<ContactPartners> encodeAcids(ArrayList<ContactPartners> strands) {
        for (int i = 0; i < strands.size(); i++) {
            String[] acids = strands.get(i).sequence.split(" ");
            String encoded_strand = "";
            for (int j = 0; j < acids.length; j++) {
                encoded_strand += acid_codes.get(acids[j]) + " ";
            }
            strands.get(i).sequence = encoded_strand.trim();
        }
        return strands;
    }


    static void generateSets(int window_size) {
        buildProteins(BASE_NAME + 0);
        ArrayList<ContactPartners> partners = findSuccesses(window_size);
        partners = encodeAcids(partners);

        String writefilename = "testset.txt";
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(writefilename));

            for (int i = 0; i < partners.size(); i++) {
                writer.write(partners.get(i).id + "\n");
                writer.write("1\n");
                writer.write(partners.get(i).sequence + "\n");
                writer.write(partners.get(i).classification + "\n\n");
            }

        } catch (java.io.IOException uee) {
            System.out.println("Could not write to file");
        }
    }

}