from eerieview.data_models import EERIEMember


def test_eerie_member():
    member_str = "ifs-fesom2-sr.hist-1950.v20240304.atmos.gr025.2D_monthly_avg"
    member = EERIEMember.from_string(member_str)
    print(member)
