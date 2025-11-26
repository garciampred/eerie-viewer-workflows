from eerieview.data_models import CmorEerieMember, EERIEMember


def test_eerie_member():
    member_str = "ifs-fesom2-sr.hist-1950.v20240304.atmos.gr025.2D_monthly_avg"
    member = EERIEMember.from_string(member_str)
    assert member.to_string() == member_str
    member_ocean = member.to_ocean()
    assert (
        member_ocean.to_string()
        == "ifs-fesom2-sr.hist-1950.v20240304.ocean.gr025.2D_monthly_avg"
    )


def test_cmor_eerie_member():
    member_str = "ifs-nemo-er.hist-1950.v20250516.gr025.Amon"
    member = CmorEerieMember.from_string(member_str)
    assert member.to_string() == member_str
    member_ocean = member.to_ocean()
    assert member_ocean.to_string() == "ifs-nemo-er.hist-1950.v20250516.gr025.Omon"
