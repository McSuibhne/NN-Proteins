import java.util.ArrayList;
//Object to store read in raw protein information before processing
class Protein {

    String name;
    int length;

    ArrayList<AminoAcid> chain = new ArrayList<>();

    Protein(String name, int length, String[] primary_structure, String[] secondary_structure, String[] partner_indexes) {
        this.name = name;
        this.length = length;

        for (int i=0; i<length; i++){
            chain.add(new AminoAcid());
        }


        for (int i = 0; i < length; i++) {
            chain.get(i).acid = primary_structure[i];
            chain.get(i).secondary_structure = secondary_structure[i];
            chain.get(i).partner_index = partner_indexes[i];
        }
    }

    ArrayList<ContactPartners> findPartners(int window_size) {
        ArrayList<ContactPartners> partners = new ArrayList<>();
        int current_middle_1 = window_size / 2;
        int current_middle_2 = current_middle_1 + 1;
        while (current_middle_1 < (length - window_size / 2 - 1)) {

            if (chain.get(current_middle_1).secondary_structure.equals("E")) {
                if (Integer.parseInt(chain.get(current_middle_1).partner_index) == current_middle_2 + 1) {

                    String strand1 = "", strand2 = "";
                    for (int i = 0; i < window_size; i++) {

                        strand1 += chain.get(current_middle_1 + i - window_size / 2).acid + " ";
                        strand2 += chain.get(current_middle_2 + i - window_size / 2).acid + " ";

                    }
                    partners.add(new ContactPartners(strand1 + strand2.trim(), name, "1"));

                }
                if (current_middle_2 == length - 1 - window_size / 2) {
                    current_middle_1++;
                    current_middle_2 = current_middle_1;
                }
                current_middle_2++;
            }

        }

        return partners;
    }
}
