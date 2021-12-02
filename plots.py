import Functions as fn
import pandas as pd
import matplotlib.pyplot as plt


full_profiles, full_edges = fn.load_and_select_profiles_and_edges("Y")
full_profiles.set_index("user_id", drop=False, inplace=True)

network = fn.create_graph_from_nodes_and_edges(full_profiles, full_edges)

edges_with_features = fn.add_node_features_to_edges(full_profiles, full_edges)

edges_with_features["type"] = edges_with_features.apply(
    lambda x: 1 if x["gender_y"] == x["gender_x"] else -1, axis=1)

male = pd.concat([fn.genderfilter(
    "MF", edges_with_features), fn.genderfilter("MM", edges_with_features)])

female = pd.concat([fn.genderfilter(
    "FF", edges_with_features), fn.genderfilter("FM", edges_with_features)])


fn.plot_degree_distribution(network)
'''
Az ábrából látható, hogy a emberek legnagyob része kevés ismerőssel rendelkezik,
a legtöbb node fokszáma alacsony, viszont van pár ember aki a többiekhez képest
nagyon sokkal(~100 nagyságrend), az eloszlás nem normális, vannak extrém, kiugró értékek.
'''

plt.figure()
fn.plot_age_distribution_by_gender(full_profiles)
'''
Az adatbázisban 15-50 éves korig vannak megfigyelések, az ábrából látható, hogy
a közödsségi portálok sokkal népszerűbbek a fiatalok körében, a férfiak vannak
többségben a 20-40 éves korosztályban, efelett már a nők, ez az eloszlás társadalmi
szinten is valamelyest megfigylehető.
'''

plt.figure()
fn.plot_node_degree_by_gender(full_profiles, network)
'''
Minimális különbség figyelhető meg az ismerősök számában a nemeket illetően,
fiatalabb korban a nőknek általában több ismerősük van, azonban 35 éves korban
változik a helyzet. Látható a korábban említett tény, hogy a fiatalok többségben
vannak, általában az emberek hasonló korosztállyal ismerősök. Az idősek ismerősei
feltehetően általában a gyerekeik és csak kevesebb idős ember.
'''

plt.figure()
fn.plot_node_average_neighbor_degree_by_gender(full_profiles, network)
'''
20 éves kor fölött a férfiak ismerőseinek átlagos fokszáma jelentősen a 
nők fölött van, a hálózaton több a férfi felhasználó, általában azonos nemből több
embert ismerünk, a baráti klikkekben is megfigyelhető női és férfi társaásgok.
A fiatalok ismét többségen vannak, több fiatal van fent a hálózaton akik ismerik
egymást, ezáltal a saját és ismerőseik átlagos fokszáma is nagyobb.
'''

plt.figure()
fn.plot_node_clustering_by_gender(full_profiles, network)
'''
A korábban említett feltételezések jelenlétét mutatja az ábra, a klaszterek mutatják,
hogy alacsonyabb korban nagyon magas, általában az osztályunkkal barátkozunk,
ahol mindenki ismer mindenkit, majd ahogy dolgozni kezdünk ez a jelenség csökken.
A férfiak fiatalabb korban többségben vannak, és jobban ismerik ismerőseik ismerőseit.
A nagyobb férfi társaságok jelenlétére utal az ábra. Idős korban nő a klaszterezettség,
ismerőseik száma kevesebb, és jobban ismerik egymást, azokkal az emberkekkel barátkozunk
hosszabb távon, akikkel régebb óta ismerjük egymást, idősebbek kevésbé törekednek új
barátságok alakítására és a kevés ismerősüket valószínűb, hogy bemutatják egymásnak hosszabb
távon.
Idősebb korban a nők klaszterezettsége nő, általában a gyerekeik barátainak szüleivel
általában az anyák tartják a kapcsolatot, így a klaszterezettségük nő.
'''

plt.figure()
fn.plot_age_relations_heatmap_genderdiff(male, "Age of male")

plt.figure()
fn.plot_age_relations_heatmap_genderdiff(female, "Age of female")
'''
A következő 2 ábrát együtt érdemes értelmezni, az ábák ellenkező nemmű ismerőseinek
része megyegyezik, különbséget az jelenti,hogy az elsőben a női-női, a másodikban a
férfi-férfi kapcsolatok vannak kiszűrve. Mindkét ábrából átható, hogy leginkább 
hasonló korú ismerőseink vannak, ami az idő előrehaladtával csökken. A nők gyakrabban
ismerősek a fiatalabb nőkkel,a férfiak általában inkább a saját korsztályukkal
ismerősök. A legtöbb ismerettség a fiatal korban jellemző, a fiatalok túslúyából
adódóan, idősebb korban a férfi-férfi barátságok száma relatíve kevés, a férfiaknak
idősebb korban csak kevés, feltehetőleg jó barátjuk van.
'''

plt.figure()
fn.plot_age_relations_heatmap(edges_with_features)

plt.figure()
fn.plot_age_relations_heatmapv2(fn.genderfilter("FF", edges_with_features), "Female-Female relation")

plt.figure()
fn.plot_age_relations_heatmapv2(fn.genderfilter("MF", edges_with_features), "Female-Male relation")

plt.figure()
fn.plot_age_relations_heatmapv2(fn.genderfilter("MM", edges_with_features), "Male-Male relation")
'''
ezeknek a heatmapeknek nem kéne szimmetrikusnak lenniük?

'''

profiles = fn.portion_separator(network, full_profiles.sample(n=30000), full_profiles)

plt.figure()
fn.portion_plot(profiles)
'''
Az utolsó ábrán látható annak megoszlása, hogy a különböző korokban milyen arányban
vannak az adott nembe és korkülönbségbe tartozó ismerősök aránya. Fiatalabb korban
lényegesen nagyobb részarányt képviselnek a hasonló korú ismerősök, ez egyrészt a 
vizsgált korosztályból is adódik, a 15 évesek általában ismernek maguknál fiatalabb
embereket, akik nem szerepelnek az elemzésben, ez igaz az idősebb korosztályra is, 
másrészt 15 évesen nem nagyon lehet olyan ismerősünk, aki 20 évvel fiatalabb nálunk.
Az idő előrehaladtával egyre több korosztályból ismerünk emberket.
'''